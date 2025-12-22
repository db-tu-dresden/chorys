/**
 * \file granforge/dbops/hash_join_functor.hpp
 * \brief Simple hash table used by the custom hash join bridge.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "tsl/tsl.hpp"

namespace chorys::granforge::dbops {

struct HashJoinRowRef {
  size_t batch_index;
  int64_t row_index;
};

template <typename KeyT>
class HashJoinTable {
 public:
  static_assert(std::is_integral_v<KeyT>,
                "HashJoinTable supports integral keys only.");

  void Add(KeyT key, size_t batch_index, int64_t row_index) {
    EnsureCapacity(1);
    auto& bucket = FindOrInsert(key);
    bucket.push_back(HashJoinRowRef{batch_index, row_index});
  }

  const std::vector<HashJoinRowRef>* Find(KeyT key) const {
    if (keys_.empty()) {
      return nullptr;
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
          auto tz = CountTrailingZeros(bits);
          size_t pos = idx + static_cast<size_t>(tz);
          if (IsOccupied(occupied_, pos)) {
            return &values_[pos];
          }
          bits &= static_cast<decltype(bits)>(bits - 1);
        }
        for (size_t offset = 0; offset < stride; ++offset) {
          size_t pos = idx + offset;
          if (!IsOccupied(occupied_, pos)) {
            return nullptr;
          }
        }
        idx = (idx + stride) & mask;
        continue;
      }

      if (!IsOccupied(occupied_, idx)) {
        return nullptr;
      }
      if (keys_[idx] == key) {
        return &values_[idx];
      }
      idx = (idx + 1) & mask;
    }
  }

  bool empty() const { return size_ == 0; }

 private:
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
    std::vector<int64_t> old_keys;
    std::vector<std::vector<HashJoinRowRef>> old_values;
    std::vector<uint64_t> old_occupied;
    old_keys.swap(keys_);
    old_values.swap(values_);
    old_occupied.swap(occupied_);

    keys_.assign(capacity, 0);
    values_.assign(capacity, {});
    occupied_.assign((capacity + 63) / 64, 0);
    size_ = 0;

    if (old_keys.empty()) {
      return;
    }
    for (size_t i = 0; i < old_keys.size(); ++i) {
      if (IsOccupied(old_occupied, i)) {
        auto& bucket = FindOrInsert(old_keys[i]);
        bucket = std::move(old_values[i]);
      }
    }
  }

  std::vector<HashJoinRowRef>& FindOrInsert(KeyT key) {
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
          auto tz = CountTrailingZeros(bits);
          size_t pos = idx + static_cast<size_t>(tz);
          if (IsOccupied(occupied_, pos)) {
            return values_[pos];
          }
          bits &= static_cast<decltype(bits)>(bits - 1);
        }
        for (size_t offset = 0; offset < stride; ++offset) {
          size_t pos = idx + offset;
          if (!IsOccupied(occupied_, pos)) {
            keys_[pos] = key;
            SetOccupied(pos);
            size_ += 1;
            return values_[pos];
          }
        }
        idx = (idx + stride) & mask;
        continue;
      }

      if (!IsOccupied(occupied_, idx)) {
        keys_[idx] = key;
        SetOccupied(idx);
        size_ += 1;
        return values_[idx];
      }
      if (keys_[idx] == key) {
        return values_[idx];
      }
      idx = (idx + 1) & mask;
    }
  }

  template <typename MaskT>
  static int CountTrailingZeros(MaskT mask) {
    using UnsignedT = std::make_unsigned_t<MaskT>;
    auto value = static_cast<UnsignedT>(mask);
    if (value == 0) {
      return static_cast<int>(std::numeric_limits<UnsignedT>::digits);
    }
    if constexpr (sizeof(UnsignedT) <= sizeof(uint32_t)) {
      return __builtin_ctz(static_cast<uint32_t>(value));
    } else {
      return __builtin_ctzll(static_cast<unsigned long long>(value));
    }
  }

  std::vector<KeyT> keys_;
  std::vector<std::vector<HashJoinRowRef>> values_;
  std::vector<uint64_t> occupied_;
  size_t size_ = 0;

  static bool IsOccupied(const std::vector<uint64_t>& mask, size_t pos) {
    size_t word = pos >> 6;
    size_t bit = pos & 63U;
    return (mask[word] >> bit) & 1ULL;
  }

  void SetOccupied(size_t pos) {
    size_t word = pos >> 6;
    size_t bit = pos & 63U;
    occupied_[word] |= (1ULL << bit);
  }
};

}  // namespace chorys::granforge::dbops
