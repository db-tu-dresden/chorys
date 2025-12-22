/**
 * \file granforge/bridges/arrow/ops/order_by.hpp
 * \brief ExecNode that sorts rows using dbops helpers to build ordering indices.
 *
 * The node buffers incoming data, validates that the sort key is supported, and once the
 * upstream pipeline finishes, it constructs a sorted batch by shuffling buffered vectors
 * with the indices produced by `dbops::BuildSortedIndices`.
 */
#pragma once

#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>
#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/exec.h>
#include <arrow/compute/ordering.h>
#include <arrow/datum.h>
#include <arrow/memory_pool.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>

#include "granforge/dbops/order_by_functor.hpp"

namespace chorys::granforge::arrow_bridge {

namespace ac = ::arrow::acero;
namespace cp = ::arrow::compute;
namespace db = chorys::granforge::dbops;

/// \brief ExecNode that implements a single-column ascending ORDER BY for integer types.
class OrderByNode : public ac::ExecNode {
 public:
  static arrow::Result<ac::ExecNode*> Make(ac::ExecPlan* plan,
                                           std::vector<ac::ExecNode*> inputs,
                                           const ac::ExecNodeOptions& options) {
    if (inputs.size() != 1) {
      return arrow::Status::Invalid(
          "OrderByNode expects exactly one input");
    }

    const auto& order_opts = static_cast<const ac::OrderByNodeOptions&>(options);
    const auto& sort_keys = order_opts.ordering.sort_keys();
    if (sort_keys.size() != 1) {
      return arrow::Status::NotImplemented(
          "OrderByNode currently supports a single sort key");
    }
    if (sort_keys[0].order != cp::SortOrder::Ascending) {
      return arrow::Status::NotImplemented(
          "OrderByNode only supports ascending order");
    }

    auto input_schema = inputs[0]->output_schema();
    ARROW_ASSIGN_OR_RAISE(auto sort_path,
                          sort_keys[0].target.FindOne(*input_schema));
    if (sort_path.indices().size() != 1) {
      return arrow::Status::Invalid("Unsupported nested sort key");
    }
    int sort_index = sort_path.indices()[0];
    if (sort_index < 0 || sort_index >= input_schema->num_fields()) {
      return arrow::Status::Invalid("Sort key index out of bounds");
    }

    auto node = std::unique_ptr<OrderByNode>(new OrderByNode(
        plan, std::move(inputs), std::move(input_schema), sort_index,
        order_opts.ordering));
    return plan->AddNode(std::move(node));
  }

  /// \brief ExecNode descriptor for explain output.
  const char* kind_name() const override { return "chorys_order_by"; }

  /// Buffer incoming column data until the sort key can be applied globally.
  arrow::Status InputReceived(ac::ExecNode*, cp::ExecBatch batch) override {
    return BufferBatch(std::move(batch));
  }

  /// Sort the buffered rows once the upstream nodes finish producing.
  arrow::Status InputFinished(ac::ExecNode*, int) override {
    ARROW_RETURN_NOT_OK(EmitSortedBatch());
    return this->output_->InputFinished(this, 1);
  }

  void PauseProducing(ac::ExecNode*, int32_t) override {}
  void ResumeProducing(ac::ExecNode*, int32_t) override {}

  arrow::Status StartProducing() override { return arrow::Status::OK(); }
  arrow::Status StopProducing() override { return StopProducingImpl(); }

 protected:
  arrow::Status StopProducingImpl() override { return arrow::Status::OK(); }

 private:
  OrderByNode(ac::ExecPlan* plan, std::vector<ac::ExecNode*> inputs,
              std::shared_ptr<arrow::Schema> output_schema, int sort_column_index,
              cp::Ordering ordering)
      : ac::ExecNode(plan, std::move(inputs), std::vector<std::string>{"input"},
                     std::move(output_schema)),
        sort_column_index_(sort_column_index),
        ordering_(std::move(ordering)) {
    auto schema = this->output_schema();
    columns_.resize(schema->num_fields());
  }

  /// \brief Materialize and buffer a batch of input rows.
  inline arrow::Status BufferBatch(cp::ExecBatch batch) {
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    ARROW_ASSIGN_OR_RAISE(auto rb,
                          batch.ToRecordBatch(this->output_schema(), pool));

    for (int i = 0; i < rb->num_columns(); ++i) {
      ARROW_RETURN_NOT_OK(AppendColumnValues(i, rb->column(i)));
    }
    return arrow::Status::OK();
  }

  /// \brief Append all values of a column array to the buffered storage.
  inline arrow::Status AppendColumnValues(
      int column_index, const std::shared_ptr<arrow::Array>& array) {
    if (array->null_count() > 0) {
      return arrow::Status::NotImplemented(
          "OrderByNode does not support null values yet");
    }
    if (array->type_id() == arrow::Type::INT64) {
      return AppendIntArray(column_index,
                            *std::static_pointer_cast<arrow::Int64Array>(array));
    }
    if (array->type_id() == arrow::Type::INT32) {
      return AppendIntArray(column_index,
                            *std::static_pointer_cast<arrow::Int32Array>(array));
    }
    return arrow::Status::NotImplemented(
        "OrderByNode only supports int32/int64 columns");
  }

  /// \brief Append int32 values to the buffered column.
  inline arrow::Status AppendIntArray(int column_index,
                                      const arrow::Int32Array& array) {
    auto& column = columns_[column_index];
    column.reserve(column.size() + static_cast<size_t>(array.length()));
    for (int64_t row = 0; row < array.length(); ++row) {
      column.push_back(array.Value(row));
    }
    return arrow::Status::OK();
  }

  /// \brief Append int64 values to the buffered column.
  inline arrow::Status AppendIntArray(int column_index,
                                      const arrow::Int64Array& array) {
    auto& column = columns_[column_index];
    column.reserve(column.size() + static_cast<size_t>(array.length()));
    for (int64_t row = 0; row < array.length(); ++row) {
      column.push_back(array.Value(row));
    }
    return arrow::Status::OK();
  }

  const cp::Ordering& ordering() const override { return ordering_; }

  /// \brief Construct and emit a sorted ExecBatch based on the buffered data.
  arrow::Status EmitSortedBatch() {
    if (emitted_) {
      return arrow::Status::OK();
    }
    emitted_ = true;

    ARROW_ASSIGN_OR_RAISE(auto sort_view, SortView());
    db::OrderByView view{sort_view};
    auto indices = db::BuildSortedIndices(view);

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(columns_.size());
    for (size_t col = 0; col < columns_.size(); ++col) {
      ARROW_ASSIGN_OR_RAISE(auto arr,
                            BuildColumn(col, indices, pool));
      arrays.push_back(std::move(arr));
    }

    auto batch = arrow::RecordBatch::Make(
        this->output_schema(), static_cast<int64_t>(indices.size()), arrays);
    cp::ExecBatch exec_batch(*batch);
    exec_batch.index = 0;
    exec_batch.guarantee = cp::literal(true);
    ARROW_RETURN_NOT_OK(
        this->output_->InputReceived(this, std::move(exec_batch)));
    return arrow::Status::OK();
  }

  /// \brief Obtain a span for the currently buffered sort key.
  inline arrow::Result<std::span<const int64_t>> SortView() const {
    if (sort_column_index_ >= static_cast<int>(columns_.size())) {
      return arrow::Status::Invalid("Sort column index out of range");
    }
    const auto& column = columns_[sort_column_index_];
    return std::span<const int64_t>(column.data(), column.size());
  }

  /// \brief Materialize a sorted column using the computed indices.
  inline arrow::Result<std::shared_ptr<arrow::Array>> BuildColumn(
      size_t column_index, const std::vector<int64_t>& indices,
      arrow::MemoryPool* pool) const {
    const auto& values = columns_[column_index];
    if (output_schema()->field(static_cast<int>(column_index))->type()->id() ==
        arrow::Type::INT64) {
      arrow::Int64Builder builder(pool);
      ARROW_RETURN_NOT_OK(builder.Reserve(indices.size()));
      for (int64_t idx : indices) {
        ARROW_RETURN_NOT_OK(
            builder.Append(values[static_cast<size_t>(idx)]));
      }
      return builder.Finish();
    }

    arrow::Int32Builder builder(pool);
    ARROW_RETURN_NOT_OK(builder.Reserve(indices.size()));
    for (int64_t idx : indices) {
      ARROW_RETURN_NOT_OK(builder.Append(
          static_cast<int32_t>(values[static_cast<size_t>(idx)])));
    }
    return builder.Finish();
  }

  int sort_column_index_;
  bool emitted_ = false;
  std::vector<std::vector<int64_t>> columns_;
  cp::Ordering ordering_;
};

}  // namespace chorys::granforge::arrow_bridge
