/**
 * \file granforge/bridges/arrow/ops/group_by.hpp
 * \brief Arrow ExecNode that implements a simplified GROUP BY with COUNT using dbops functors.
 *
 * This bridge consumes ExecBatches produced by upstream nodes, extracts the grouping column,
 * forwards it to the dbops accumulator, and emits a final RecordBatch with key/count columns.
 * The node deliberately supports only the subset required by the demo pipeline (single INT key
 * and COUNT aggregate), trading generality for transparency and easier experimentation.
 */
#pragma once

#include <memory>
#include <algorithm>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>
#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/expression.h>
#include <arrow/compute/exec.h>
#include <arrow/datum.h>
#include <arrow/memory_pool.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/util/parallel.h>

#include "granforge/dbops/group_by_functor.hpp"

namespace chorys::granforge::arrow_bridge {

namespace ac = ::arrow::acero;
namespace cp = ::arrow::compute;
namespace db = chorys::granforge::dbops;

/// \brief Description of a single-key COUNT aggregation.
struct GroupBySpec {
  int key_index;
  std::shared_ptr<arrow::Field> key_field;
  std::shared_ptr<arrow::Schema> output_schema;
  arrow::Type::type key_type;

  /// \brief Build the spec from Arrow aggregate options.
  static arrow::Result<GroupBySpec> FromAggregate(
      ac::ExecNode* input, const ac::AggregateNodeOptions& opts);
};

/// \brief Validate that the aggregate options match the supported subset.
inline arrow::Status ValidateAggregateOptions(
    const ac::AggregateNodeOptions& opts) {
  if (!opts.segment_keys.empty()) {
    return arrow::Status::NotImplemented(
        "GroupByNode does not support segment keys");
  }
  if (opts.keys.size() != 1) {
    return arrow::Status::NotImplemented(
        "GroupByNode currently supports a single grouping key");
  }
  if (opts.aggregates.size() != 1) {
    return arrow::Status::NotImplemented(
        "GroupByNode expects exactly one aggregate measure");
  }
  const auto& agg = opts.aggregates[0];
  if (agg.function.find("count") == std::string::npos) {
    return arrow::Status::NotImplemented(
        "GroupByNode only supports count aggregates");
  }
  return arrow::Status::OK();
}

/// \brief Resolve schema information for a supported aggregate declaration.
inline arrow::Result<GroupBySpec> GroupBySpec::FromAggregate(
    ac::ExecNode* input, const ac::AggregateNodeOptions& opts) {
  ARROW_RETURN_NOT_OK(ValidateAggregateOptions(opts));

  auto schema = input->output_schema();
  spdlog::info("[granforge][group_by] FieldRef: {}", opts.keys[0].ToString());

  ARROW_ASSIGN_OR_RAISE(auto key_path, opts.keys[0].FindOne(*schema));
  if (key_path.indices().size() != 1) {
    return arrow::Status::Invalid("Unsupported nested key for GroupByNode");
  }
  int key_idx = key_path.indices()[0];
  if (key_idx < 0 || key_idx >= schema->num_fields()) {
    return arrow::Status::Invalid("GroupByNode key index out of bounds");
  }

  auto key_field = schema->field(key_idx);
  if (key_field->type()->id() != arrow::Type::INT32 &&
      key_field->type()->id() != arrow::Type::INT64) {
    return arrow::Status::NotImplemented(
        "GroupByNode only supports INT32/INT64 keys");
  }
  spdlog::info("[granforge][group_by] key column: {}", key_field->name());

  const auto& agg = opts.aggregates[0];
  std::string count_name = agg.name.empty() ? agg.function : agg.name;
  auto count_field = arrow::field(count_name, arrow::int64());

  GroupBySpec spec;
  spec.key_index = key_idx;
  spec.key_field = key_field;
  spec.output_schema = arrow::schema({key_field, count_field});
  spec.key_type = key_field->type()->id();
  return spec;
}

/// \brief ExecNode that aggregates a single integer key and COUNT measure.
class GroupByNode : public ac::ExecNode {
 public:
  /// \brief Instantiate the node and resolve the grouping configuration.
  static arrow::Result<ac::ExecNode*> Make(ac::ExecPlan* plan,
                                           std::vector<ac::ExecNode*> inputs,
                                           const ac::ExecNodeOptions& options) {
    if (inputs.size() != 1) {
      return arrow::Status::Invalid(
          "GroupByNode expects exactly one input");
    }

    const auto& agg_opts = static_cast<const ac::AggregateNodeOptions&>(options);
    ARROW_ASSIGN_OR_RAISE(auto spec,
                          GroupBySpec::FromAggregate(inputs[0], agg_opts));

    auto node = std::unique_ptr<GroupByNode>(new GroupByNode(
        plan, std::move(inputs), std::move(spec),
        plan->query_context()->executor()));
    return plan->AddNode(std::move(node));
  }

  /// \brief ExecNode descriptor for explain output.
  const char* kind_name() const override { return "chorys_group_by"; }

  /// Consume a batch of data, extract the key column, and feed it to the accumulator.
  arrow::Status InputReceived(ac::ExecNode*, cp::ExecBatch batch) override {
    return ConsumeKeys(batch);
  }

  /// Emit the aggregated batch once the upstream pipeline indicates completion.
  arrow::Status InputFinished(ac::ExecNode*, int) override {
    ARROW_RETURN_NOT_OK(EmitResult());
    return this->output_->InputFinished(this, 1);
  }

  void PauseProducing(ac::ExecNode*, int32_t) override {}
  void ResumeProducing(ac::ExecNode*, int32_t) override {}

  arrow::Status StartProducing() override { return arrow::Status::OK(); }

  arrow::Status StopProducing() override { return StopProducingImpl(); }

 protected:
  arrow::Status StopProducingImpl() override { return arrow::Status::OK(); }

 private:
  GroupByNode(ac::ExecPlan* plan, std::vector<ac::ExecNode*> inputs,
              GroupBySpec spec, ::arrow::internal::Executor* executor)
      : ac::ExecNode(plan, std::move(inputs), std::vector<std::string>{"input"},
                     spec.output_schema),
        spec_(std::move(spec)),
        executor_(executor) {
    if (spec_.key_type == arrow::Type::INT32) {
      accumulator_.emplace<db::GroupByAccumulator<int32_t>>();
    } else {
      accumulator_.emplace<db::GroupByAccumulator<int64_t>>();
    }
  }

  /// \brief Buffer the key values from an ExecBatch and feed them to dbops.
  arrow::Status ConsumeKeys(const cp::ExecBatch& batch) {
    if (spec_.key_index >= static_cast<int>(batch.values.size())) {
      return arrow::Status::Invalid("GroupByNode key index out of range");
    }

    const arrow::Datum& datum = batch.values[spec_.key_index];
    if (!datum.is_array()) {
      return arrow::Status::Invalid(
          "GroupByNode expects array inputs for keys");
    }

    auto array = datum.make_array();
    if (array->null_count() > 0) {
      return arrow::Status::NotImplemented(
          "GroupByNode does not support null keys yet");
    }

    if (array->type_id() == arrow::Type::INT64) {
      const auto& ints = static_cast<const arrow::Int64Array&>(*array);
      auto keys = std::span<const int64_t>(ints.raw_values(), ints.length());
      auto& acc = std::get<db::GroupByAccumulator<int64_t>>(accumulator_);
      return ConsumeKeysParallel(keys, &acc);
    }

    const auto& ints = static_cast<const arrow::Int32Array&>(*array);
    auto keys = std::span<const int32_t>(ints.raw_values(), ints.length());
    auto& acc = std::get<db::GroupByAccumulator<int32_t>>(accumulator_);
    return ConsumeKeysParallel(keys, &acc);
  }

  /// \brief Materialize a RecordBatch holding the aggregated keys and counts.
  arrow::Status EmitResult() {
    if (emitted_) {
      return arrow::Status::OK();
    }
    emitted_ = true;

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    return std::visit(
        [&](auto& acc) -> arrow::Status {
          auto result = acc.Finish();
          int64_t length = static_cast<int64_t>(result.keys.size());

          ARROW_ASSIGN_OR_RAISE(auto key_array,
                                MakeKeyColumn(result.keys, pool));
          ARROW_ASSIGN_OR_RAISE(auto count_array,
                                MakeCountColumn(result.counts, pool));

          std::vector<std::shared_ptr<arrow::Array>> columns = {key_array,
                                                                count_array};
          auto batch =
              arrow::RecordBatch::Make(this->output_schema(), length, columns);
          cp::ExecBatch exec_batch(*batch);
          exec_batch.index = 0;
          exec_batch.guarantee = cp::literal(true);
          ARROW_RETURN_NOT_OK(
              this->output_->InputReceived(this, std::move(exec_batch)));
          return arrow::Status::OK();
        },
        accumulator_);
  }

  /// \brief Materialize the grouped key column for the output batch.
  template <typename KeyT>
  arrow::Result<std::shared_ptr<arrow::Array>> MakeKeyColumn(
      const std::vector<KeyT>& keys, arrow::MemoryPool* pool) {
    if constexpr (std::is_same_v<KeyT, int64_t>) {
      arrow::Int64Builder builder(pool);
      ARROW_RETURN_NOT_OK(builder.AppendValues(keys));
      return builder.Finish();
    } else {
      static_assert(std::is_same_v<KeyT, int32_t>,
                    "GroupByNode only supports int32/int64 keys");
      arrow::Int32Builder builder(pool);
      ARROW_RETURN_NOT_OK(builder.AppendValues(keys));
      return builder.Finish();
    }
  }

  /// \brief Materialize the grouped count column for the output batch.
  arrow::Result<std::shared_ptr<arrow::Array>> MakeCountColumn(
      const std::vector<int64_t>& counts, arrow::MemoryPool* pool) {
    arrow::Int64Builder builder(pool);
    ARROW_RETURN_NOT_OK(builder.AppendValues(counts));
    return builder.Finish();
  }

  template <typename KeyT>
  arrow::Status ConsumeKeysParallel(
      std::span<const KeyT> keys, db::GroupByAccumulator<KeyT>* acc) {
    const int64_t length = static_cast<int64_t>(keys.size());
    if (length == 0) {
      return arrow::Status::OK();
    }

    const int64_t chunk_size = 1 << 14;
    int64_t task_count = (length + chunk_size - 1) / chunk_size;
    if (executor_) {
      task_count = std::min<int64_t>(task_count, executor_->GetCapacity());
    }
    int num_tasks = static_cast<int>(std::max<int64_t>(1, task_count));
    if (!executor_ || num_tasks == 1) {
      db::GroupByView<KeyT> view{keys};
      acc->Consume(view);
      return arrow::Status::OK();
    }

    std::vector<db::GroupByAccumulator<KeyT>> locals(num_tasks);
    auto status = ::arrow::internal::OptionalParallelFor(
        executor_ && num_tasks > 1, num_tasks,
        [&](int task_id) -> arrow::Status {
          int64_t start = static_cast<int64_t>(task_id) * chunk_size;
          int64_t end = std::min(length, start + chunk_size);
          if (start >= end) {
            return arrow::Status::OK();
          }
          db::GroupByView<KeyT> view{keys.subspan(
              static_cast<size_t>(start),
              static_cast<size_t>(end - start))};
          locals[static_cast<size_t>(task_id)].Consume(view);
          return arrow::Status::OK();
        },
        executor_);
    ARROW_RETURN_NOT_OK(status);

    for (int i = 0; i < num_tasks; ++i) {
      acc->MergeFrom(locals[static_cast<size_t>(i)]);
    }
    return arrow::Status::OK();
  }

  GroupBySpec spec_;
  std::variant<db::GroupByAccumulator<int32_t>,
               db::GroupByAccumulator<int64_t>> accumulator_;
  ::arrow::internal::Executor* executor_ = nullptr;
  bool emitted_ = false;
}; 

}  // namespace chorys::granforge::arrow_bridge
