/**
 * \file granforge/dbops/group_by_functor.hpp
 * \brief Minimal accumulator used by the custom Arrow group-by node.
 *
 * The struct is intentionally tiny: it only supports a single INT key and COUNT aggregate,
 * which keeps the accumulator easy to reason about when debugging Substrait plans.
 */
#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include <spdlog/spdlog.h>

#include "tsl/tsl.hpp"

namespace chorys::granforge::dbops {

/// \brief Span describing a batch of grouping keys to be consumed.
template <typename KeyT>
struct GroupByView {
  std::span<const KeyT> keys;
};

/// \brief Final aggregation result emitted by GroupByAccumulator.
template <typename KeyT>
struct GroupByResult {
  std::vector<KeyT> keys;
  std::vector<int64_t> counts;
};

/// \brief Order-preserving COUNT accumulator for a single grouping column.
template <typename KeyT>
class GroupByAccumulator {
 public:
  /// Consume a new batch of keys, updating internal counts and insertion order.
  void Consume(const GroupByView<KeyT>& view) {
    spdlog::info("[granforge][group_by] consuming {} keys", view.keys.size());
    EnsureCapacity(view.keys.size());
    for (KeyT key : view.keys) {
      InsertOrAdd(key, 1, true);
    }
  }

  /// Produce the ordered key/count vectors describing the aggregated state.
  GroupByResult<KeyT> Finish() const {
    GroupByResult<KeyT> result;
    result.keys.reserve(order_.size());
    result.counts.reserve(order_.size());
    for (KeyT key : order_) {
      auto count = FindCount(key);
      if (count != 0) {
        result.keys.push_back(key);
        result.counts.push_back(count);
      }
    }
    return result;
  }

  /// Merge another accumulator into this one, preserving first-seen order.
  void MergeFrom(const GroupByAccumulator& other) {
    if (other.size_ == 0) {
      return;
    }
    EnsureCapacity(other.size_);
    for (KeyT key : other.order_) {
      auto count = other.FindCount(key);
      if (count != 0) {
        InsertOrAdd(key, count, true);
      }
    }
  }

  /// Reset the accumulator so it can be reused for another group-by operation.
  void Reset() {
    counts_.clear();
    order_.clear();
    keys_.clear();
    size_ = 0;
  }

 private:
  static_assert(std::is_integral_v<KeyT>,
                "GroupByAccumulator currently supports integral keys only.");

  static constexpr double kMaxLoadFactor = 0.7;
  static constexpr size_t kInitialCapacity = 1024;

  static size_t NextPow2(size_t value) {
    if (value == 0) {
      return 1;
    }
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    if constexpr (sizeof(size_t) >= 8) {
      value |= value >> 32;
    }
    return value + 1;
  }

  static size_t HashKey(KeyT key) {
    uint64_t x = static_cast<uint64_t>(key);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<size_t>(x);
  }

  void EnsureCapacity(size_t incoming) {
    if (keys_.empty()) {
      Reserve(NextPow2(std::max(kInitialCapacity, incoming * 2)));
      return;
    }
    size_t needed = size_ + incoming;
    size_t capacity = keys_.size();
    if (static_cast<double>(needed) >= capacity * kMaxLoadFactor) {
      Reserve(capacity * 2);
    }
  }

  void Reserve(size_t capacity) {
    capacity = NextPow2(capacity);
    std::vector<KeyT> old_keys;
    std::vector<int64_t> old_counts;
    old_keys.swap(keys_);
    old_counts.swap(counts_);
    keys_.assign(capacity, KeyT{});
    counts_.assign(capacity, 0);
    size_ = 0;
    if (old_keys.empty()) {
      return;
    }
    for (size_t i = 0; i < old_keys.size(); ++i) {
      if (old_counts[i] != 0) {
        InsertOrAdd(old_keys[i], old_counts[i], false);
      }
    }
  }

  void InsertOrAdd(KeyT key, int64_t delta, bool record_order) {
    if (keys_.empty()) {
      Reserve(kInitialCapacity);
    }
    size_t capacity = keys_.size();
    size_t mask = capacity - 1;
    size_t idx = HashKey(key) & mask;

    using Extension = tsl::runtime::cpu::max_width_extension_t;
    using Vec = tsl::simd<KeyT, Extension>;
    const size_t stride = Vec::vector_element_count();
    auto needle = tsl::set1<Vec>(key);

    for (;;) {
      if (idx + stride <= capacity) {
        auto key_vec = tsl::loadu<Vec>(keys_.data() + idx);
        auto eq_mask = tsl::equal<Vec>(key_vec, needle);
        auto bits = tsl::to_integral<Vec>(eq_mask);
        while (bits) {
          auto tz = tsl::tzc<Vec>(bits);
          size_t pos = idx + static_cast<size_t>(tz);
          if (counts_[pos] != 0) {
            counts_[pos] += delta;
            return;
          }
          bits &= static_cast<decltype(bits)>(bits - 1);
        }
        for (size_t offset = 0; offset < stride; ++offset) {
          size_t pos = idx + offset;
          if (counts_[pos] == 0) {
            keys_[pos] = key;
            counts_[pos] = delta;
            size_ += 1;
            if (record_order) {
              order_.push_back(key);
            }
            return;
          }
        }
        idx = (idx + stride) & mask;
        continue;
      }

      if (counts_[idx] == 0) {
        keys_[idx] = key;
        counts_[idx] = delta;
        size_ += 1;
        if (record_order) {
          order_.push_back(key);
        }
        return;
      }
      if (keys_[idx] == key) {
        counts_[idx] += delta;
        return;
      }
      idx = (idx + 1) & mask;
    }
  }

  int64_t FindCount(KeyT key) const {
    if (keys_.empty()) {
      return 0;
    }
    size_t capacity = keys_.size();
    size_t mask = capacity - 1;
    size_t idx = HashKey(key) & mask;

    using Extension = tsl::runtime::cpu::max_width_extension_t;
    using Vec = tsl::simd<KeyT, Extension>;
    const size_t stride = Vec::vector_element_count();
    auto needle = tsl::set1<Vec>(key);

    for (;;) {
      if (idx + stride <= capacity) {
        auto key_vec = tsl::loadu<Vec>(keys_.data() + idx);
        auto eq_mask = tsl::equal<Vec>(key_vec, needle);
        auto bits = tsl::to_integral<Vec>(eq_mask);
        while (bits) {
          auto tz = tsl::tzc<Vec>(bits);
          size_t pos = idx + static_cast<size_t>(tz);
          if (counts_[pos] != 0) {
            return counts_[pos];
          }
          bits &= static_cast<decltype(bits)>(bits - 1);
        }
        for (size_t offset = 0; offset < stride; ++offset) {
          size_t pos = idx + offset;
          if (counts_[pos] == 0) {
            return 0;
          }
        }
        idx = (idx + stride) & mask;
        continue;
      }

      if (counts_[idx] == 0) {
        return 0;
      }
      if (keys_[idx] == key) {
        return counts_[idx];
      }
      idx = (idx + 1) & mask;
    }
  }

  std::vector<KeyT> keys_;
  std::vector<int64_t> counts_;
  size_t size_ = 0;
  std::vector<KeyT> order_;
};

}  // namespace chorys::granforge::dbops
