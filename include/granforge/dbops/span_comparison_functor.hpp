/**
 * \file granforge/dbops/span_comparison_functor.hpp
 * \brief SIMD helpers that evaluate simple comparison predicates over spans of integers.
 *
 * These utilities power the "span comparison" fast-path that can filter batches without
 * invoking the full Arrow expression engine, useful for benchmarking custom kernels.
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

/// \brief Enumeration of comparison operators supported by the span functors.
enum class ComparisonOp {
  kLess,
  kLessEqual,
  kGreater,
  kGreaterEqual,
  kEqual,
  kNotEqual
};

/// \brief Description of a single comparison to evaluate.
template <typename ValueT>
struct SpanComparison {
  static_assert(std::is_arithmetic_v<ValueT>,
                "SpanComparison requires an arithmetic value type.");
  ComparisonOp op;
  ValueT value;
};

/// \brief Bundle describing the values to inspect and comparisons to apply.
template <typename ValueT>
struct SpanComparisonView {
  static_assert(std::is_arithmetic_v<ValueT>,
                "SpanComparisonView requires an arithmetic value type.");
  std::span<const ValueT> values;
  std::span<const SpanComparison<ValueT>> comparisons;
};

template <typename ValueT>
struct EvaluateComparisonsFn {
  static_assert(std::is_arithmetic_v<ValueT>,
                "EvaluateComparisonsFn requires an arithmetic value type.");
  constexpr EvaluateComparisonsFn() noexcept = default;

  /// \brief Return the indices that pass every predicate in \p view.
  std::vector<int64_t> operator()(const SpanComparisonView<ValueT>& view) const {
    spdlog::info("[granforge][span_compare] evaluating {} rows with {} predicates",
                 view.values.size(), view.comparisons.size());
    std::vector<int64_t> indices;
    indices.reserve(view.values.size());
    using Extension = tsl::runtime::cpu::max_width_extension_t;
    using Vec = tsl::simd<ValueT, Extension>;
    constexpr size_t kStride = Vec::vector_element_count();
    const auto all_true = tsl::to_mask<Vec>(tsl::integral_all_true<Vec>());
    size_t i = 0;
    while (i + kStride <= view.values.size()) {
      auto vals = tsl::loadu<Vec>(view.values.data() + i);
      auto mask = all_true;
      for (const auto& cmp : view.comparisons) {
        auto cmp_val = tsl::set1<Vec>(cmp.value);
        switch (cmp.op) {
          case ComparisonOp::kLess:
            mask = tsl::mask_binary_and<Vec>(
                mask, tsl::less_than<Vec>(vals, cmp_val));
            break;
          case ComparisonOp::kLessEqual: {
            mask = tsl::mask_binary_and<Vec>(
                mask, tsl::less_than_or_equal<Vec>(vals, cmp_val));
            break;
          }
          case ComparisonOp::kGreater:
            mask = tsl::mask_binary_and<Vec>(
                mask, tsl::greater_than<Vec>(vals, cmp_val));
            break;
          case ComparisonOp::kGreaterEqual: {
            mask = tsl::mask_binary_and<Vec>(
                mask, tsl::greater_than_or_equal<Vec>(vals, cmp_val));
            break;
          }
          case ComparisonOp::kEqual:
            mask = tsl::mask_binary_and<Vec>(
                mask, tsl::equal<Vec>(vals, cmp_val));
            break;
          case ComparisonOp::kNotEqual: {
            mask = tsl::mask_binary_and<Vec>(
                mask, tsl::nequal<Vec>(vals, cmp_val));
            break;
          }
        }
      }
      auto bits = tsl::to_integral<Vec>(mask);
      while (bits) {
        auto bit = CountTrailingZeros(bits);
        indices.push_back(static_cast<int64_t>(i + static_cast<size_t>(bit)));
        bits &= static_cast<decltype(bits)>(bits - 1);
      }
      i += kStride;
    }
    for (; i < view.values.size(); ++i) {
      bool ok = true;
      for (const auto& cmp : view.comparisons) {
        if (!Compare(view.values[i], cmp)) {
          ok = false;
          break;
        }
      }
      if (ok) {
        indices.push_back(static_cast<int64_t>(i));
      }
    }
    return indices;
  }

 private:
  static bool Compare(ValueT value, const SpanComparison<ValueT>& cmp) {
    switch (cmp.op) {
      case ComparisonOp::kLess:
        return value < cmp.value;
      case ComparisonOp::kLessEqual:
        return value <= cmp.value;
      case ComparisonOp::kGreater:
        return value > cmp.value;
      case ComparisonOp::kGreaterEqual:
        return value >= cmp.value;
      case ComparisonOp::kEqual:
        return value == cmp.value;
      case ComparisonOp::kNotEqual:
        return value != cmp.value;
    }
    return false;
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
};

template <typename ValueT>
inline constexpr EvaluateComparisonsFn<ValueT> EvaluateComparisons{};

}  // namespace chorys::granforge::dbops
