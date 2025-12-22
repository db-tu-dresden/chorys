/**
 * \file granforge/bridges/arrow/ops/filter.hpp
 * \brief Custom Arrow Acero filter node that delegates predicate evaluation to dbops functors.
 *
 * The bridge now exposes a single node that chooses the best filtering strategy at construction
 * time (either a bound Arrow expression or the dbops span comparison kernel).  The runtime cost
 * of virtual dispatch/factory indirection disappears and the code flow is compact enough to read
 * without jumping between helper classes.
 */
#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/util.h>
#include <arrow/acero/query_context.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/acero/map_node.h>
#include <arrow/compute/api_vector.h>
#include <arrow/compute/expression.h>
#include <arrow/compute/exec.h>
#include <arrow/datum.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>
#include <arrow/util/parallel.h>

#include "granforge/dbops/filter_mask_functor.hpp"
#include "granforge/dbops/span_comparison_functor.hpp"

namespace chorys::granforge::arrow_bridge {

namespace db = chorys::granforge::dbops;
using db::ComparisonOp;
using db::EvaluateFilterMask;
using db::FilterMaskView;

namespace ac = ::arrow::acero;
namespace cp = ::arrow::compute;

namespace detail {
template <typename T>
bool TryCastScalarValue(const arrow::Scalar& scalar, T* out) {
  switch (scalar.type->id()) {
    case arrow::Type::INT8:
      *out = static_cast<T>(static_cast<const arrow::Int8Scalar&>(scalar).value);
      return true;
    case arrow::Type::INT16:
      *out = static_cast<T>(static_cast<const arrow::Int16Scalar&>(scalar).value);
      return true;
    case arrow::Type::INT32:
      *out = static_cast<T>(static_cast<const arrow::Int32Scalar&>(scalar).value);
      return true;
    case arrow::Type::INT64:
      *out = static_cast<T>(static_cast<const arrow::Int64Scalar&>(scalar).value);
      return true;
    case arrow::Type::UINT8:
      *out = static_cast<T>(static_cast<const arrow::UInt8Scalar&>(scalar).value);
      return true;
    case arrow::Type::UINT16:
      *out = static_cast<T>(static_cast<const arrow::UInt16Scalar&>(scalar).value);
      return true;
    case arrow::Type::UINT32:
      *out = static_cast<T>(static_cast<const arrow::UInt32Scalar&>(scalar).value);
      return true;
    case arrow::Type::UINT64:
      *out = static_cast<T>(static_cast<const arrow::UInt64Scalar&>(scalar).value);
      return true;
    case arrow::Type::FLOAT:
      *out = static_cast<T>(static_cast<const arrow::FloatScalar&>(scalar).value);
      return true;
    case arrow::Type::DOUBLE:
      *out = static_cast<T>(static_cast<const arrow::DoubleScalar&>(scalar).value);
      return true;
    default:
      return false;
  }
}
}  // namespace detail

/// \brief ExecNode options that describe which strategy the filter node should use.
/// \brief Options object describing which filter strategy should be instantiated.
struct FilterOptions : public ac::ExecNodeOptions {
  struct ComparisonLiteral {
    ComparisonOp op;
    std::shared_ptr<arrow::Scalar> value;
  };

  struct SpanComparisonSpec {
    arrow::FieldRef field_ref;
    std::vector<std::vector<ComparisonLiteral>> clauses;
  };

  std::variant<std::monostate, cp::Expression, SpanComparisonSpec> spec;

  /// \brief Create options that wrap an Arrow expression.
  static FilterOptions FromExpression(cp::Expression expr);
  /// \brief Create options that wrap a span-comparison descriptor.
  static FilterOptions FromComparisons(arrow::FieldRef field_ref,
                                       std::vector<std::vector<ComparisonLiteral>> clauses);
};

/// \brief Arrow ExecNode that rewrites filter declarations into calls into dbops functors.
///
/// Each instance owns a concrete filtering strategy resolved from FilterOptions.  When batches
/// arrive from upstream nodes the active strategy is invoked to materialize the filtered result.
class FilterNode : public ac::MapNode {
 public:
  /// \brief Instantiate the node and resolve the filtering strategy.
  static arrow::Result<ac::ExecNode*> Make(ac::ExecPlan* plan,
                                           std::vector<ac::ExecNode*> inputs,
                                           const ac::ExecNodeOptions& options);

  /// \brief ExecNode descriptor used when producing textual plans.
  const char* kind_name() const override { return "chorys_filter"; }

 protected:
  /// \brief Invoke the chosen strategy on the incoming batch.
  arrow::Result<cp::ExecBatch> ProcessBatch(cp::ExecBatch batch) override;

 private:
  struct ExpressionStrategy {
    /// \brief Bind an Arrow expression to the provided schema.
    static arrow::Result<ExpressionStrategy> Bind(
        cp::Expression expr, const std::shared_ptr<arrow::Schema>& schema);

    /// \brief Evaluate the expression against the incoming batch.
    arrow::Result<cp::ExecBatch> operator()(cp::ExecBatch batch,
                                            ::arrow::internal::Executor* executor);

   private:
    /// \brief Create an empty batch when the predicate is known to be false.
    arrow::Result<cp::ExecBatch> MakeEmptyBatch(
        int64_t index, const cp::Expression& guarantee);

    std::shared_ptr<arrow::Schema> schema_;
    cp::Expression bound_expression_;
  };

  struct SpanComparisonStrategy {
    /// \brief Configure the span-comparison strategy for the provided schema.
    static arrow::Result<SpanComparisonStrategy> Bind(
        const std::shared_ptr<arrow::Schema>& schema,
        const FilterOptions::SpanComparisonSpec& spec);

    /// \brief Run span comparisons across the incoming batch.
    arrow::Result<cp::ExecBatch> operator()(cp::ExecBatch batch,
                                            ::arrow::internal::Executor* executor);

    int column_index_;
    std::shared_ptr<arrow::Schema> schema_;
    std::vector<std::vector<FilterOptions::ComparisonLiteral>> clauses_;

    template <typename T>
    arrow::Result<cp::ExecBatch> ApplyComparisons(
        const std::shared_ptr<arrow::Array>& array, cp::ExecBatch batch,
        ::arrow::internal::Executor* executor) const;
  };

  using Strategy = std::variant<ExpressionStrategy, SpanComparisonStrategy>;

  FilterNode(ac::ExecPlan* plan, std::vector<ac::ExecNode*> inputs,
             std::shared_ptr<arrow::Schema> output_schema, Strategy strategy,
             ::arrow::internal::Executor* executor)
      : ac::MapNode(plan, std::move(inputs), std::move(output_schema)),
        strategy_(std::move(strategy)),
        executor_(executor) {}

  Strategy strategy_;
  ::arrow::internal::Executor* executor_;
};

inline FilterOptions FilterOptions::FromExpression(cp::Expression expr) {
  FilterOptions opts;
  opts.spec = std::move(expr);
  return opts;
}

inline FilterOptions FilterOptions::FromComparisons(
    arrow::FieldRef field_ref,
    std::vector<std::vector<FilterOptions::ComparisonLiteral>> clauses) {
  FilterOptions opts;
  FilterOptions::SpanComparisonSpec spec{
      .field_ref = std::move(field_ref),
      .clauses = std::move(clauses),
  };
  opts.spec = std::move(spec);
  return opts;
}

inline arrow::Result<ac::ExecNode*> FilterNode::Make(
    ac::ExecPlan* plan, std::vector<ac::ExecNode*> inputs,
    const ac::ExecNodeOptions& options) {
  if (inputs.size() != 1) {
    return arrow::Status::Invalid("FilterNode expects exactly one input");
  }

  const auto& filter_opts = static_cast<const FilterOptions&>(options);
  auto schema = inputs[0]->output_schema();

  auto make_strategy = [&]() -> arrow::Result<Strategy> {
    if (auto expr = std::get_if<cp::Expression>(&filter_opts.spec)) {
      return ExpressionStrategy::Bind(*expr, schema);
    }
    if (auto spec =
            std::get_if<FilterOptions::SpanComparisonSpec>(&filter_opts.spec)) {
      return SpanComparisonStrategy::Bind(schema, *spec);
    }
    return arrow::Status::Invalid(
        "FilterOptions did not contain a valid strategy");
  };

  ARROW_ASSIGN_OR_RAISE(auto strategy, make_strategy());

  auto node = std::unique_ptr<FilterNode>(new FilterNode(
      plan, std::move(inputs), std::move(schema), std::move(strategy),
      plan->query_context()->executor()));
  return plan->AddNode(std::move(node));
}

inline arrow::Result<cp::ExecBatch> FilterNode::ProcessBatch(
    cp::ExecBatch batch) {
  return std::visit(
      [&](auto& strategy) -> arrow::Result<cp::ExecBatch> {
        return strategy(std::move(batch), executor_);
      },
      strategy_);
}

inline arrow::Result<FilterNode::ExpressionStrategy>
FilterNode::ExpressionStrategy::Bind(
    cp::Expression expr, const std::shared_ptr<arrow::Schema>& schema) {
  cp::ExecContext exec_ctx(arrow::default_memory_pool());
  ARROW_ASSIGN_OR_RAISE(auto bound, expr.Bind(*schema, &exec_ctx));
  ExpressionStrategy strategy;
  strategy.schema_ = schema;
  strategy.bound_expression_ = std::move(bound);
  return strategy;
}

inline arrow::Result<cp::ExecBatch>
FilterNode::ExpressionStrategy::operator()(cp::ExecBatch batch,
                                           ::arrow::internal::Executor* executor) {
  cp::ExecContext exec_ctx(arrow::default_memory_pool());
  ARROW_ASSIGN_OR_RAISE(auto predicate,
                        cp::ExecuteScalarExpression(bound_expression_, batch,
                                                    &exec_ctx));

  // Fast-path when the predicate collapses to a scalar (all rows same result).
  if (predicate.is_scalar()) {
    const auto& scalar = predicate.scalar_as<arrow::BooleanScalar>();
    // Invalid or false scalar means "keep nothing", so return an empty batch.
    if (!scalar.is_valid || !scalar.value) {
      return MakeEmptyBatch(batch.index, batch.guarantee);
    }
    // True scalar means "keep everything", so reuse the input batch.
    return batch;
  }

  // Only array predicates are meaningful for row-wise filtering.
  if (predicate.kind() != arrow::Datum::ARRAY) {
    return arrow::Status::Invalid(
        "Filter predicate must yield boolean array or scalar");
  }

  auto predicate_array = predicate.make_array();
  auto bool_array =
      std::static_pointer_cast<arrow::BooleanArray>(predicate_array);
  // Build a byte mask (0xFF = keep, 0x00 = drop) from the boolean array.
  std::vector<uint8_t> mask(bool_array->length());
  const int64_t length = bool_array->length();
  const int64_t chunk_size = 1 << 14;
  int64_t task_count = (length + chunk_size - 1) / chunk_size;
  if (executor) {
    // Respect executor capacity so we do not oversubscribe.
    task_count = std::min<int64_t>(task_count, executor->GetCapacity());
  }
  int num_tasks = static_cast<int>(std::max<int64_t>(1, task_count));
  auto fill_status = ::arrow::internal::OptionalParallelFor(
      executor && num_tasks > 1, num_tasks,
      [&](int task_id) -> arrow::Status {
        int64_t start = task_id * chunk_size;
        int64_t end = std::min(length, start + chunk_size);
        for (int64_t i = start; i < end; ++i) {
          mask[static_cast<size_t>(i)] =
              (bool_array->IsValid(i) && bool_array->Value(i)) ? 0xFF : 0x00;
        }
        return arrow::Status::OK();
      },
      executor);
  ARROW_RETURN_NOT_OK(fill_status);

  // Convert the byte mask into a list of selected row indices.
  std::vector<int64_t> selected;
  if (!executor || num_tasks == 1) {
    FilterMaskView view{std::span<const uint8_t>(mask.data(), mask.size())};
    selected = EvaluateFilterMask(view);
  } else {
    std::vector<std::vector<int64_t>> locals(
        static_cast<size_t>(num_tasks));
    auto select_status = ::arrow::internal::OptionalParallelFor(
        executor && num_tasks > 1, num_tasks,
        [&](int task_id) -> arrow::Status {
          int64_t start = task_id * chunk_size;
          int64_t end = std::min(length, start + chunk_size);
          if (start >= end) {
            return arrow::Status::OK();
          }
          FilterMaskView view{std::span<const uint8_t>(
              mask.data() + start, static_cast<size_t>(end - start))};
          auto local = EvaluateFilterMask(view);
          for (auto& idx : local) {
            idx += start;
          }
          locals[static_cast<size_t>(task_id)] = std::move(local);
          return arrow::Status::OK();
        },
        executor);
    ARROW_RETURN_NOT_OK(select_status);

    size_t total = 0;
    for (const auto& local : locals) {
      total += local.size();
    }
    selected.reserve(total);
    for (auto& local : locals) {
      selected.insert(selected.end(), local.begin(), local.end());
    }
  }

  arrow::Int64Builder index_builder;
  ARROW_RETURN_NOT_OK(index_builder.AppendValues(selected));
  std::shared_ptr<arrow::Int64Array> indices;
  ARROW_RETURN_NOT_OK(index_builder.Finish(&indices));

  // Materialize a RecordBatch so we can Take() each column by indices.
  ARROW_ASSIGN_OR_RAISE(
      auto batch_record,
      batch.ToRecordBatch(schema_, exec_ctx.memory_pool()));

  std::vector<std::shared_ptr<arrow::Array>> filtered_columns;
  filtered_columns.reserve(batch_record->num_columns());

  for (int i = 0; i < batch_record->num_columns(); ++i) {
    // Apply the same row selection to each column.
    ARROW_ASSIGN_OR_RAISE(
        auto taken,
        cp::Take(arrow::Datum(batch_record->column(i)),
                 arrow::Datum(indices),
                 cp::TakeOptions::Defaults(), &exec_ctx));
    filtered_columns.push_back(taken.make_array());
  }

  // Rebuild the batch with filtered columns and preserve exec metadata.
  auto filtered_batch =
      arrow::RecordBatch::Make(schema_, indices->length(), filtered_columns);
  cp::ExecBatch out(*filtered_batch);
  out.index = batch.index;
  out.guarantee = batch.guarantee;
  return out;
}

inline arrow::Result<cp::ExecBatch>
FilterNode::ExpressionStrategy::MakeEmptyBatch(
    int64_t index, const cp::Expression& guarantee) {
  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(schema_->num_fields());
  for (const auto& field : schema_->fields()) {
    ARROW_ASSIGN_OR_RAISE(auto arr,
                          arrow::MakeArrayOfNull(field->type(), /*length=*/0));
    columns.push_back(std::move(arr));
  }
  auto empty_batch = arrow::RecordBatch::Make(schema_, 0, std::move(columns));
  cp::ExecBatch out(*empty_batch);
  out.index = index;
  out.guarantee = guarantee;
  return out;
}

inline arrow::Result<FilterNode::SpanComparisonStrategy>
FilterNode::SpanComparisonStrategy::Bind(
    const std::shared_ptr<arrow::Schema>& schema,
    const FilterOptions::SpanComparisonSpec& spec) {
  ARROW_ASSIGN_OR_RAISE(auto path, spec.field_ref.FindOne(*schema));
  if (path.indices().size() != 1) {
    return arrow::Status::Invalid(
        "Span comparison filter expects a flat field reference");
  }
  return SpanComparisonStrategy{
      .column_index_ = path.indices()[0],
      .schema_ = schema,
      .clauses_ = spec.clauses,
  };
}

inline arrow::Result<cp::ExecBatch>
FilterNode::SpanComparisonStrategy::operator()(cp::ExecBatch batch,
                                               ::arrow::internal::Executor* executor) {
  if (column_index_ >= static_cast<int>(batch.values.size())) {
    return arrow::Status::Invalid("Column index out of range for filter batch");
  }

  const arrow::Datum& datum = batch.values[column_index_];
  if (!datum.is_array()) {
    return arrow::Status::Invalid(
        "SpanComparisonFilter expects array inputs");
  }

  auto array = datum.make_array();
  if (array->null_count() > 0) {
    return arrow::Status::NotImplemented(
        "Span comparison filter does not support nulls yet");
  }
  switch (array->type_id()) {
    case arrow::Type::INT8:
      return ApplyComparisons<int8_t>(array, std::move(batch), executor);
    case arrow::Type::INT16:
      return ApplyComparisons<int16_t>(array, std::move(batch), executor);
    case arrow::Type::INT32:
      return ApplyComparisons<int32_t>(array, std::move(batch), executor);
    case arrow::Type::INT64:
      return ApplyComparisons<int64_t>(array, std::move(batch), executor);
    case arrow::Type::UINT8:
      return ApplyComparisons<uint8_t>(array, std::move(batch), executor);
    case arrow::Type::UINT16:
      return ApplyComparisons<uint16_t>(array, std::move(batch), executor);
    case arrow::Type::UINT32:
      return ApplyComparisons<uint32_t>(array, std::move(batch), executor);
    case arrow::Type::UINT64:
      return ApplyComparisons<uint64_t>(array, std::move(batch), executor);
    case arrow::Type::FLOAT:
      return ApplyComparisons<float>(array, std::move(batch), executor);
    case arrow::Type::DOUBLE:
      return ApplyComparisons<double>(array, std::move(batch), executor);
    default:
      return arrow::Status::NotImplemented(
          "Span comparison filter only supports arithmetic columns");
  }

}
template <typename T>
arrow::Result<cp::ExecBatch> FilterNode::SpanComparisonStrategy::ApplyComparisons(
    const std::shared_ptr<arrow::Array>& array,
    cp::ExecBatch batch, ::arrow::internal::Executor* executor) const {
  using ArrayT = typename arrow::TypeTraits<typename arrow::CTypeTraits<T>::ArrowType>::ArrayType;
  auto typed = std::static_pointer_cast<ArrayT>(array);
  auto values = std::span<const T>(typed->raw_values(), typed->length());
  std::vector<uint8_t> selected_flags(values.size(), 0);
  const int64_t length = static_cast<int64_t>(values.size());
  const int64_t chunk_size = 1 << 14;
  int64_t task_count = (length + chunk_size - 1) / chunk_size;
  if (executor) {
    task_count = std::min<int64_t>(task_count, executor->GetCapacity());
  }
  int num_tasks = static_cast<int>(std::max<int64_t>(1, task_count));
  for (const auto& clause : clauses_) {
    std::vector<db::SpanComparison<T>> comparisons;
    comparisons.reserve(clause.size());
    for (const auto& cmp : clause) {
      if (!cmp.value) {
        return arrow::Status::Invalid("Span comparison literal is null");
      }
      T value{};
      if (!detail::TryCastScalarValue<T>(*cmp.value, &value)) {
        return arrow::Status::Invalid(
            "Span comparison literal type is not numeric");
      }
      comparisons.push_back(db::SpanComparison<T>{cmp.op, value});
    }
    db::SpanComparisonView<T> view{
        values,
        std::span<const db::SpanComparison<T>>(comparisons.data(),
                                               comparisons.size())};
    if (!executor || num_tasks == 1) {
      auto selected = db::EvaluateComparisons<T>(view);
      for (int64_t index : selected) {
        selected_flags[static_cast<size_t>(index)] = 1;
      }
    } else {
      std::vector<std::vector<int64_t>> locals(
          static_cast<size_t>(num_tasks));
      auto status = ::arrow::internal::OptionalParallelFor(
          executor && num_tasks > 1, num_tasks,
          [&](int task_id) -> arrow::Status {
            int64_t start = task_id * chunk_size;
            int64_t end = std::min(length, start + chunk_size);
            if (start >= end) {
              return arrow::Status::OK();
            }
            db::SpanComparisonView<T> task_view{
                values.subspan(static_cast<size_t>(start),
                               static_cast<size_t>(end - start)),
                std::span<const db::SpanComparison<T>>(comparisons.data(),
                                                       comparisons.size())};
            auto local = db::EvaluateComparisons<T>(task_view);
            for (auto& idx : local) {
              idx += start;
            }
            locals[static_cast<size_t>(task_id)] = std::move(local);
            return arrow::Status::OK();
          },
          executor);
      ARROW_RETURN_NOT_OK(status);
      for (const auto& local : locals) {
        for (int64_t index : local) {
          selected_flags[static_cast<size_t>(index)] = 1;
        }
      }
    }
  }

  std::vector<int64_t> selected;
  selected.reserve(values.size());
  for (size_t i = 0; i < selected_flags.size(); ++i) {
    if (selected_flags[i]) {
      selected.push_back(static_cast<int64_t>(i));
    }
  }

  arrow::Int64Builder index_builder;
  ARROW_RETURN_NOT_OK(index_builder.AppendValues(selected));
  std::shared_ptr<arrow::Int64Array> indices;
  ARROW_RETURN_NOT_OK(index_builder.Finish(&indices));

  cp::ExecContext exec_ctx(arrow::default_memory_pool());
  ARROW_ASSIGN_OR_RAISE(
      auto batch_record,
      batch.ToRecordBatch(schema_, exec_ctx.memory_pool()));

  std::vector<std::shared_ptr<arrow::Array>> filtered_columns;
  filtered_columns.reserve(batch_record->num_columns());

  for (int i = 0; i < batch_record->num_columns(); ++i) {
    ARROW_ASSIGN_OR_RAISE(
        auto taken,
        cp::Take(arrow::Datum(batch_record->column(i)),
                 arrow::Datum(indices),
                 cp::TakeOptions::Defaults(), &exec_ctx));
    filtered_columns.push_back(taken.make_array());
  }

  auto filtered_batch =
      arrow::RecordBatch::Make(schema_, indices->length(), filtered_columns);
  cp::ExecBatch out(*filtered_batch);
  out.index = batch.index;
  out.guarantee = batch.guarantee;
  return out;
}

struct ComparisonDescriptor {
  arrow::FieldRef field_ref;
  std::vector<std::vector<FilterOptions::ComparisonLiteral>> clauses;
  bool has_field = false;
};

inline bool ParseComparisonCall(const cp::Expression::Call& call,
                                ComparisonDescriptor* desc,
                                std::vector<FilterOptions::ComparisonLiteral>* out) {
  auto parse_cmp = [](const std::string& fn_name)
      -> std::optional<ComparisonOp> {
    if (fn_name == "less") return ComparisonOp::kLess;
    if (fn_name == "less_equal") return ComparisonOp::kLessEqual;
    if (fn_name == "greater") return ComparisonOp::kGreater;
    if (fn_name == "greater_equal") return ComparisonOp::kGreaterEqual;
    if (fn_name == "equal") return ComparisonOp::kEqual;
    if (fn_name == "not_equal") return ComparisonOp::kNotEqual;
    return std::nullopt;
  };

  auto cmp_op = parse_cmp(call.function_name);
  if (!cmp_op) {
    return false;
  }
  if (call.arguments.size() != 2) {
    return false;
  }

  const cp::Expression* field_expr = nullptr;
  const cp::Expression* literal_expr = nullptr;
  for (const auto& arg : call.arguments) {
    if (arg.field_ref()) {
      field_expr = &arg;
    } else if (arg.literal()) {
      literal_expr = &arg;
    }
  }
  if (!field_expr || !literal_expr) {
    return false;
  }

  if (!desc->has_field) {
    desc->field_ref = *field_expr->field_ref();
    desc->has_field = true;
  } else if (!desc->field_ref.Equals(*field_expr->field_ref())) {
    return false;
  }

  const arrow::Datum* lit = literal_expr->literal();
  if (!lit || !lit->is_scalar()) {
    return false;
  }
  auto scalar = lit->scalar();
  if (!scalar) {
    return false;
  }
  switch (scalar->type->id()) {
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      break;
    default:
      return false;
  }
  out->push_back(FilterOptions::ComparisonLiteral{*cmp_op, scalar});
  return true;
}

inline bool ExtractClause(const cp::Expression& expr,
                          ComparisonDescriptor* desc,
                          std::vector<FilterOptions::ComparisonLiteral>* out) {
  if (const auto* call = expr.call()) {
    if (call->function_name == "and" || call->function_name == "and_kleene") {
      for (const auto& arg : call->arguments) {
        if (!ExtractClause(arg, desc, out)) {
          return false;
        }
      }
      return !out->empty();
    }
    return ParseComparisonCall(*call, desc, out);
  }
  return false;
}

inline bool ExtractOrComparisons(const cp::Expression& expr,
                                 ComparisonDescriptor* desc) {
  if (const auto* call = expr.call();
      call && (call->function_name == "or" ||
               call->function_name == "or_kleene")) {
    for (const auto& arg : call->arguments) {
      std::vector<FilterOptions::ComparisonLiteral> clause;
      if (!ExtractClause(arg, desc, &clause)) {
        return false;
      }
      desc->clauses.push_back(std::move(clause));
    }
    return !desc->clauses.empty();
  }

  std::vector<FilterOptions::ComparisonLiteral> clause;
  if (!ExtractClause(expr, desc, &clause)) {
    return false;
  }
  desc->clauses.push_back(std::move(clause));
  return true;
}

/// \brief Attempt to build span-comparison options from an Arrow expression.
inline std::optional<FilterOptions> MaybeMakeSpanComparisonOptions(
    const cp::Expression& expr) {
  ComparisonDescriptor desc;
  if (!ExtractOrComparisons(expr, &desc)) {
    return std::nullopt;
  }
  return FilterOptions::FromComparisons(desc.field_ref,
                                        std::move(desc.clauses));
}

}  // namespace chorys::granforge::arrow_bridge
