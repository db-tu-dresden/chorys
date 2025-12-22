/**
 * \file granforge/bridges/arrow/ops/hash_join.hpp
 * \brief ExecNode implementing an INT32/INT64 equi hash join using dbops helpers.
 */
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>
#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/exec.h>
#include <arrow/datum.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>

#include <spdlog/spdlog.h>

#include "granforge/dbops/hash_join_functor.hpp"

namespace chorys::granforge::arrow_bridge {

namespace ac = ::arrow::acero;
namespace cp = ::arrow::compute;
namespace db = chorys::granforge::dbops;

/// \brief Description of an equi-join configuration.
struct HashJoinSpec {
  std::shared_ptr<arrow::Schema> left_schema;
  std::shared_ptr<arrow::Schema> right_schema;
  std::shared_ptr<arrow::Schema> output_schema;
  int left_key_index = -1;
  int right_key_index = -1;

  /// \brief Build a join spec from Arrow HashJoin options.
  static arrow::Result<HashJoinSpec> Make(
      const std::vector<ac::ExecNode*>& inputs,
      const ac::HashJoinNodeOptions& options);
};

/// \brief Validate that the join declaration matches the supported subset.
inline arrow::Status ValidateJoinOptions(
    const ac::HashJoinNodeOptions& options) {
  if (options.join_type != ac::JoinType::INNER) {
    return arrow::Status::NotImplemented(
        "HashJoinNode currently supports only INNER joins");
  }
  if (options.left_keys.size() != 1 || options.right_keys.size() != 1) {
    return arrow::Status::NotImplemented(
        "HashJoinNode currently supports a single join key");
  }
  return arrow::Status::OK();
}

/// \brief Resolve join schema and key information.
inline arrow::Result<HashJoinSpec> HashJoinSpec::Make(
    const std::vector<ac::ExecNode*>& inputs,
    const ac::HashJoinNodeOptions& options) {
  if (inputs.size() != 2) {
    return arrow::Status::Invalid(
        "HashJoinNode expects exactly two inputs");
  }

  ARROW_RETURN_NOT_OK(ValidateJoinOptions(options));

  auto left_schema = inputs[0]->output_schema();
  auto right_schema = inputs[1]->output_schema();

  ARROW_ASSIGN_OR_RAISE(auto left_path,
                        options.left_keys[0].FindOne(*left_schema));
  ARROW_ASSIGN_OR_RAISE(auto right_path,
                        options.right_keys[0].FindOne(*right_schema));

  if (left_path.indices().size() != 1 ||
      right_path.indices().size() != 1) {
    return arrow::Status::Invalid(
        "HashJoinNode does not support nested key references");
  }

  int left_key_index = left_path.indices()[0];
  int right_key_index = right_path.indices()[0];

  auto left_field = left_schema->field(left_key_index);
  auto right_field = right_schema->field(right_key_index);
  auto ensure_int = [](const std::shared_ptr<arrow::Field>& field)
                        -> arrow::Status {
        if (field->type()->id() == arrow::Type::INT32 ||
            field->type()->id() == arrow::Type::INT64) {
          return arrow::Status::OK();
        }
        return arrow::Status::NotImplemented(
            "HashJoinNode only supports INT32/INT64 keys");
      };
  ARROW_RETURN_NOT_OK(ensure_int(left_field));
  ARROW_RETURN_NOT_OK(ensure_int(right_field));

  std::vector<std::shared_ptr<arrow::Field>> fields =
      left_schema->fields();
  fields.insert(fields.end(),
                right_schema->fields().begin(),
                right_schema->fields().end());
  HashJoinSpec spec;
  spec.left_schema = left_schema;
  spec.right_schema = right_schema;
  spec.output_schema = arrow::schema(fields);
  spec.left_key_index = left_key_index;
  spec.right_key_index = right_key_index;
  return spec;
}

class HashJoinNode : public ac::ExecNode {
 public:
  /// \brief Instantiate the hash join node with validated options.
  static arrow::Result<ac::ExecNode*> Make(ac::ExecPlan* plan,
                                           std::vector<ac::ExecNode*> inputs,
                                           const ac::ExecNodeOptions& options);

  const char* kind_name() const override { return "chorys_hash_join"; }

  /// \brief Buffer incoming batches from either side of the join.
  arrow::Status InputReceived(ac::ExecNode* input,
                              cp::ExecBatch batch) override;

  /// \brief Track completion of each input and emit joined rows.
  arrow::Status InputFinished(ac::ExecNode* input, int32_t total_batches) override;

  void PauseProducing(ac::ExecNode*, int32_t) override {}
  void ResumeProducing(ac::ExecNode*, int32_t) override {}

  /// \brief Begin buffering batches from upstream inputs.
  arrow::Status StartProducing() override { return arrow::Status::OK(); }
  /// \brief Stop production and release any temporary state.
  arrow::Status StopProducing() override { return StopProducingImpl(); }

 protected:
  arrow::Status StopProducingImpl() override { return arrow::Status::OK(); }

 private:
  struct StoredBatch {
    std::shared_ptr<arrow::RecordBatch> batch;
    std::vector<int64_t> keys;
  };

  HashJoinNode(ac::ExecPlan* plan, std::vector<ac::ExecNode*> inputs,
               HashJoinSpec spec);

  arrow::Status ReceiveBatch(bool is_left, cp::ExecBatch batch);

  /// \brief Materialize batches and key vectors from ExecBatches.
  arrow::Result<StoredBatch> MakeStoredBatch(
      cp::ExecBatch batch, const std::shared_ptr<arrow::Schema>& schema,
      int key_index);

  /// \brief Convert an ExecBatch to a RecordBatch with the expected schema.
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> ToRecordBatch(
      cp::ExecBatch batch, const std::shared_ptr<arrow::Schema>& schema);

  /// \brief Extract the join keys from a RecordBatch column.
  arrow::Result<std::vector<int64_t>> ExtractKeys(
      const arrow::Array& column);

  /// \brief Read a single scalar value from a buffered array.
  arrow::Result<int64_t> ExtractValue(
      const std::shared_ptr<arrow::Array>& array, int64_t row);

  arrow::Status BuildHashTable();
  arrow::Status EmitJoined();

  arrow::Status AppendRowValues(const StoredBatch& stored, int64_t row,
                                std::vector<std::vector<int64_t>>* columns,
                                size_t column_offset);

  arrow::Status BuildOutput(
      const std::vector<std::vector<int64_t>>& columns);

  ac::ExecNode* left_input_;
  ac::ExecNode* right_input_;
  HashJoinSpec spec_;
  arrow::MemoryPool* pool_;
  bool left_finished_ = false;
  bool right_finished_ = false;
  bool hash_ready_ = false;
  std::vector<StoredBatch> left_batches_;
  std::vector<StoredBatch> right_batches_;
  db::HashJoinTable<int64_t> hash_table_;
};

inline HashJoinNode::HashJoinNode(
    ac::ExecPlan* plan, std::vector<ac::ExecNode*> inputs,
    HashJoinSpec spec)
    : ac::ExecNode(plan, std::move(inputs),
                   std::vector<std::string>{"left", "right"},
                   spec.output_schema),
      left_input_(this->inputs_[0]),
      right_input_(this->inputs_[1]),
      spec_(std::move(spec)),
      pool_(arrow::default_memory_pool()) {}

inline arrow::Result<ac::ExecNode*> HashJoinNode::Make(
    ac::ExecPlan* plan, std::vector<ac::ExecNode*> inputs,
    const ac::ExecNodeOptions& options) {
  const auto& join_opts =
      static_cast<const ac::HashJoinNodeOptions&>(options);
  ARROW_ASSIGN_OR_RAISE(auto spec,
                        HashJoinSpec::Make(inputs, join_opts));

  auto node = std::unique_ptr<HashJoinNode>(new HashJoinNode(
      plan, std::move(inputs), std::move(spec)));
  return plan->AddNode(std::move(node));
}

inline arrow::Status HashJoinNode::InputReceived(ac::ExecNode* input,
                                                 cp::ExecBatch batch) {
  bool is_left = (input == left_input_);
  return ReceiveBatch(is_left, std::move(batch));
}

/// \brief Buffer an ExecBatch from either the build or probe side.
inline arrow::Status HashJoinNode::ReceiveBatch(bool is_left,
                                                cp::ExecBatch batch) {
  const auto& schema = is_left ? spec_.left_schema : spec_.right_schema;
  int key_index = is_left ? spec_.left_key_index : spec_.right_key_index;
  ARROW_ASSIGN_OR_RAISE(auto stored,
                        MakeStoredBatch(std::move(batch), schema, key_index));
  spdlog::info("[granforge][hash_join] consuming {} keys", stored.keys.size());
  if (is_left) {
    left_batches_.push_back(std::move(stored));
  } else {
    right_batches_.push_back(std::move(stored));
  }
  return arrow::Status::OK();
}

inline arrow::Result<HashJoinNode::StoredBatch> HashJoinNode::MakeStoredBatch(
    cp::ExecBatch batch, const std::shared_ptr<arrow::Schema>& schema,
    int key_index) {
  ARROW_ASSIGN_OR_RAISE(auto rb, ToRecordBatch(batch, schema));
  ARROW_ASSIGN_OR_RAISE(auto keys, ExtractKeys(*rb->column(key_index)));
  return StoredBatch{std::move(rb), std::move(keys)};
}

inline arrow::Result<std::shared_ptr<arrow::RecordBatch>>
HashJoinNode::ToRecordBatch(
    cp::ExecBatch batch, const std::shared_ptr<arrow::Schema>& schema) {
  ARROW_ASSIGN_OR_RAISE(auto rb, batch.ToRecordBatch(schema, pool_));
  return rb;
}

inline arrow::Result<std::vector<int64_t>> HashJoinNode::ExtractKeys(
    const arrow::Array& column) {
  std::vector<int64_t> keys(column.length());
  auto copy_values = [&](const auto& array) -> arrow::Status {
    for (int64_t i = 0; i < array.length(); ++i) {
      keys[static_cast<size_t>(i)] = array.Value(i);
    }
    return arrow::Status::OK();
  };

  switch (column.type_id()) {
    case arrow::Type::INT32: {
      ARROW_RETURN_NOT_OK(
          copy_values(static_cast<const arrow::Int32Array&>(column)));
      return keys;
    }
    case arrow::Type::INT64: {
      ARROW_RETURN_NOT_OK(
          copy_values(static_cast<const arrow::Int64Array&>(column)));
      return keys;
    }
    default:
      return arrow::Status::NotImplemented(
          "HashJoinNode only supports INT32/INT64 keys");
  }
}

inline arrow::Result<int64_t> HashJoinNode::ExtractValue(
    const std::shared_ptr<arrow::Array>& array, int64_t row) {
  switch (array->type_id()) {
    case arrow::Type::INT32:
      return static_cast<int64_t>(
          std::static_pointer_cast<arrow::Int32Array>(array)->Value(row));
    case arrow::Type::INT64:
      return std::static_pointer_cast<arrow::Int64Array>(array)->Value(row);
    default:
      return arrow::Status::NotImplemented(
          "HashJoinNode only supports INT32/INT64 columns");
  }
}

inline arrow::Status HashJoinNode::InputFinished(ac::ExecNode* input,
                                                 int32_t) {
  if (input == left_input_) {
    left_finished_ = true;
  } else if (input == right_input_) {
    right_finished_ = true;
  }

  if (left_finished_ && right_finished_) {
    ARROW_RETURN_NOT_OK(EmitJoined());
    return this->output_->InputFinished(this, 1);
  }
  return arrow::Status::OK();
}

/// \brief Build the hash table from buffered right-side batches.
inline arrow::Status HashJoinNode::BuildHashTable() {
  if (hash_ready_) {
    return arrow::Status::OK();
  }
  for (size_t batch_idx = 0; batch_idx < right_batches_.size(); ++batch_idx) {
    const auto& stored = right_batches_[batch_idx];
    for (int64_t row = 0; row < stored.batch->num_rows(); ++row) {
      hash_table_.Add(stored.keys[static_cast<size_t>(row)], batch_idx, row);
    }
  }
  hash_ready_ = true;
  return arrow::Status::OK();
}

/// \brief Probes the hash table with buffered left rows and emits joined output.
inline arrow::Status HashJoinNode::EmitJoined() {
  ARROW_RETURN_NOT_OK(BuildHashTable());
  if (hash_table_.empty()) {
    return BuildOutput({});
  }

  size_t total_columns = static_cast<size_t>(output_schema()->num_fields());
  std::vector<std::vector<int64_t>> columns(total_columns);

  size_t left_columns = static_cast<size_t>(spec_.left_schema->num_fields());

  for (const auto& left_batch : left_batches_) {
    for (int64_t row = 0; row < left_batch.batch->num_rows(); ++row) {
      int64_t key = left_batch.keys[static_cast<size_t>(row)];
      const auto* matches = hash_table_.Find(key);
      if (!matches) {
        continue;
      }
      for (const auto& ref : *matches) {
        const auto& right_batch = right_batches_[ref.batch_index];
        ARROW_RETURN_NOT_OK(
            AppendRowValues(left_batch, row, &columns, 0));
        ARROW_RETURN_NOT_OK(
            AppendRowValues(right_batch, ref.row_index, &columns,
                            left_columns));
      }
    }
  }

  return BuildOutput(columns);
}

/// \brief Append all column values for a specific row into the output buffers.
inline arrow::Status HashJoinNode::AppendRowValues(
    const StoredBatch& stored, int64_t row,
    std::vector<std::vector<int64_t>>* columns, size_t column_offset) {
  for (int i = 0; i < stored.batch->num_columns(); ++i) {
    auto array = stored.batch->column(i);
    ARROW_ASSIGN_OR_RAISE(auto value, ExtractValue(array, row));
    (*columns)[column_offset + static_cast<size_t>(i)].push_back(value);
  }
  return arrow::Status::OK();
}

/// \brief Materialize a RecordBatch and emit it downstream.
inline arrow::Status HashJoinNode::BuildOutput(
    const std::vector<std::vector<int64_t>>& columns) {
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  arrays.reserve(output_schema()->num_fields());
  int64_t length = columns.empty() ? 0
                                   : static_cast<int64_t>(columns[0].size());

  for (int i = 0; i < output_schema()->num_fields(); ++i) {
    auto type = output_schema()->field(i)->type();
    if (type->id() == arrow::Type::INT64) {
      arrow::Int64Builder builder(pool_);
      if (!columns.empty()) {
        ARROW_RETURN_NOT_OK(builder.AppendValues(columns[i]));
      }
      ARROW_ASSIGN_OR_RAISE(auto array, builder.Finish());
      arrays.push_back(std::move(array));
    } else if (type->id() == arrow::Type::INT32) {
      arrow::Int32Builder builder(pool_);
      if (!columns.empty()) {
        for (int64_t value : columns[i]) {
          ARROW_RETURN_NOT_OK(
              builder.Append(static_cast<int32_t>(value)));
        }
      }
      ARROW_ASSIGN_OR_RAISE(auto array, builder.Finish());
      arrays.push_back(std::move(array));
    } else {
      return arrow::Status::NotImplemented(
          "HashJoinNode only supports INT32/INT64 columns");
    }
  }

  auto batch = arrow::RecordBatch::Make(output_schema(), length, arrays);
  cp::ExecBatch exec_batch(*batch);
  exec_batch.index = 0;
  exec_batch.guarantee = cp::literal(true);
  ARROW_RETURN_NOT_OK(
      this->output_->InputReceived(this, std::move(exec_batch)));
  return arrow::Status::OK();
}

}  // namespace chorys::granforge::arrow_bridge
