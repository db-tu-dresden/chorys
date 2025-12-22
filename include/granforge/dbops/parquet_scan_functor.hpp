/**
 * \file granforge/dbops/parquet_scan_functor.hpp
 * \brief Abstract interface used to pull record batches from Parquet sources.
 */
#pragma once

#include <memory>
#include <optional>
#include <string>

namespace chorys::granforge::dbops {

/// \brief Logical description of a Parquet data source.
struct ParquetScanConfig {
  std::string path;
};

/// \brief Generic functor that yields batches of type \c BatchResult.
template <typename BatchResult>
class ParquetScanFunctor {
 public:
  using ResultType = BatchResult;
  virtual ~ParquetScanFunctor() = default;

  /// \brief Retrieve the next batch from the source.  Returning empty indicates EOF.
  virtual BatchResult NextBatch() = 0;
};

}  // namespace chorys::granforge::dbops
