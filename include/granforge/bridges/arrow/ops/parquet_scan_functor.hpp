/**
 * \file granforge/bridges/arrow/ops/parquet_scan_functor.hpp
 * \brief Adaptor that feeds dbops' Parquet scanner into Arrow ExecBatches.
 *
 * The dbops layer provides a templated ParquetScanFunctor that pushes batches through a callback.
 * This wrapper instantiates it with Arrow ExecBatch semantics so it can be wired into Acero.
 */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <arrow/compute/exec.h>
#include <arrow/io/api.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>

#include "granforge/dbops/parquet_scan_functor.hpp"

namespace chorys::granforge::arrow_bridge {

namespace cp = arrow::compute;
namespace db = chorys::granforge::dbops;

/// \brief Arrow-friendly Parquet scan functor that yields ExecBatches.
class ParquetScanFunctor
    : public db::ParquetScanFunctor<
          arrow::Result<std::optional<cp::ExecBatch>>> {
 public:
  using Config = db::ParquetScanConfig;

  /// \brief Instantiate an Arrow-friendly scan functor for the provided config.
  static arrow::Result<std::unique_ptr<ParquetScanFunctor>> Make(
      Config config);

  /// \brief Schema of the materialized RecordBatches.
  const std::shared_ptr<arrow::Schema>& schema() const { return schema_; }

  /// Fetch the next RecordBatch from the Parquet file and wrap it in an ExecBatch.
  arrow::Result<std::optional<cp::ExecBatch>> NextBatch() override;

 private:
  ParquetScanFunctor(std::shared_ptr<arrow::io::RandomAccessFile> input,
                     std::unique_ptr<parquet::arrow::FileReader> file_reader,
                     std::unique_ptr<arrow::RecordBatchReader> reader,
                     std::shared_ptr<arrow::Schema> schema)
      : input_(std::move(input)),
        file_reader_(std::move(file_reader)),
        reader_(std::move(reader)),
        schema_(std::move(schema)),
        next_batch_index_(0) {}

  std::shared_ptr<arrow::io::RandomAccessFile> input_;
  std::unique_ptr<parquet::arrow::FileReader> file_reader_;
  std::unique_ptr<arrow::RecordBatchReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  int64_t next_batch_index_;
};

/// \brief Open a Parquet file and construct an ExecBatch-producing functor.
inline arrow::Result<std::unique_ptr<ParquetScanFunctor>>
ParquetScanFunctor::Make(Config config) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();

  ARROW_ASSIGN_OR_RAISE(auto input,
                        arrow::io::ReadableFile::Open(config.path, pool));

  ARROW_ASSIGN_OR_RAISE(auto parquet_reader,
                        parquet::arrow::OpenFile(std::move(input), pool));

  ARROW_ASSIGN_OR_RAISE(auto batch_reader,
                        parquet_reader->GetRecordBatchReader());

  auto schema = batch_reader->schema();

  return std::unique_ptr<ParquetScanFunctor>(
      new ParquetScanFunctor(std::move(input), std::move(parquet_reader),
                             std::move(batch_reader), std::move(schema)));
}

/// \brief Retrieve the next ExecBatch from the underlying reader.
inline arrow::Result<std::optional<cp::ExecBatch>>
ParquetScanFunctor::NextBatch() {
  std::shared_ptr<arrow::RecordBatch> rb;
  ARROW_RETURN_NOT_OK(reader_->ReadNext(&rb));
  if (!rb) {
    return std::nullopt;
  }

  cp::ExecBatch batch(*rb);
  batch.index = next_batch_index_++;
  return batch;
}

}  // namespace chorys::granforge::arrow_bridge
