/**
 * \file granforge/dbops/order_by_functor.hpp
 * \brief Helpers that compute ordering indices for the custom ORDER BY node.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

#include <spdlog/spdlog.h>

namespace chorys::granforge::dbops {

/// \brief Immutable span backing a single-column ORDER BY.
struct OrderByView {
  std::span<const int64_t> keys;
};

struct BuildSortedIndicesFn {
  constexpr BuildSortedIndicesFn() noexcept = default;

  /// \brief Build a vector of indices describing the ascending order of \p view.
  std::vector<int64_t> operator()(const OrderByView& view) const {
    spdlog::info("[granforge][order_by] sorting {} rows", view.keys.size());
    std::vector<int64_t> indices(view.keys.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
                     [&view](int64_t lhs, int64_t rhs) {
                       return view.keys[static_cast<size_t>(lhs)] <
                              view.keys[static_cast<size_t>(rhs)];
                     });
    return indices;
  }
};

inline constexpr BuildSortedIndicesFn BuildSortedIndices{};

}  // namespace chorys::granforge::dbops
