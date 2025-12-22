/**
 * \file granforge/dbops/filter_mask_functor.hpp
 * \brief SIMD-friendly helper that translates boolean masks into row indices.
 *
 * Arrow expressions and the custom span-comparison kernels both produce byte masks that flag
 * rows to keep.  Acero expects a vector of row indices, so this helper provides the glue logic
 * (with a fast AVX2 specialization) between those two representations.
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

/// \brief Lightweight view describing a contiguous mask.
template <typename MaskT = uint8_t>
struct FilterMaskView {
  static_assert(std::is_arithmetic_v<MaskT>,
                "FilterMaskView requires an arithmetic mask type.");
  std::span<const MaskT> mask;
};

template <typename MaskT = uint8_t>
struct EvaluateFilterMaskFn {
  static_assert(std::is_arithmetic_v<MaskT>,
                "EvaluateFilterMaskFn requires an arithmetic mask type.");
  constexpr EvaluateFilterMaskFn() noexcept = default;

  /// \brief Return the positions of every non-zero byte in \p view.
  std::vector<int64_t> operator()(FilterMaskView<MaskT> view) const {
    spdlog::info("[granforge][filter_mask] evaluating {} entries", view.mask.size());
    std::vector<int64_t> indices;
    indices.reserve(view.mask.size());
    using Extension = tsl::runtime::cpu::max_width_extension_t;
    using Vec = tsl::simd<MaskT, Extension>;
    constexpr size_t kStride = Vec::vector_element_count();
    auto zero = tsl::set_zero<Vec>();
    size_t i = 0;
    while (i + kStride <= view.mask.size()) {
      auto chunk = tsl::loadu<Vec>(view.mask.data() + i);
      auto neq_mask = tsl::nequal<Vec>(chunk, zero);
      auto bits = tsl::to_integral<Vec>(neq_mask);
      while (bits) {
        auto tz = CountTrailingZeros(bits);
        indices.push_back(static_cast<int64_t>(i + static_cast<size_t>(tz)));
        bits &= static_cast<decltype(bits)>(bits - 1);
      }
      i += kStride;
    }
    for (; i < view.mask.size(); ++i) {
      if (view.mask[i] != static_cast<MaskT>(0)) {
        indices.push_back(static_cast<int64_t>(i));
      }
    }
    return indices;
  }

 private:
  template <typename MaskTLocal>
  static int CountTrailingZeros(MaskTLocal mask) {
    using UnsignedT = std::make_unsigned_t<MaskTLocal>;
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

/// \brief Customization point object for filter mask evaluation.
inline constexpr EvaluateFilterMaskFn<uint8_t> EvaluateFilterMask{};

}  // namespace chorys::granforge::dbops
