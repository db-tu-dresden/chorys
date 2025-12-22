/**
 * \file granforge/bridges/arrow/substrait/plan_rewrite.hpp
 * \brief Helpers that rewrite Substrait-derived plans to custom Acero nodes.
 */
#pragma once

#include <memory>
#include <variant>

#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>
#include <arrow/status.h>

#include "granforge/bridges/arrow/ops/filter.hpp"

namespace chorys::granforge::substrait_bridge {

namespace ac = ::arrow::acero;

/// \brief Recursively rewrite Acero declarations so selected nodes use the custom factories.
inline arrow::Status RewritePlan(ac::Declaration* decl) {
  for (auto& input : decl->inputs) {
    if (std::holds_alternative<ac::Declaration>(input)) {
      ARROW_RETURN_NOT_OK(
          RewritePlan(&std::get<ac::Declaration>(input)));
    }
  }

  if (decl->factory_name == "filter") {
    auto arrow_filter_opts =
        std::dynamic_pointer_cast<ac::FilterNodeOptions>(decl->options);
    if (!arrow_filter_opts) {
      return arrow::Status::Invalid(
          "Expected FilterNodeOptions for filter declaration");
    }

    auto span_opts =
        chorys::granforge::arrow_bridge::MaybeMakeSpanComparisonOptions(
            arrow_filter_opts->filter_expression);

    using FilterOptions = chorys::granforge::arrow_bridge::FilterOptions;
    std::shared_ptr<FilterOptions> custom_opts;
    if (span_opts) {
      custom_opts =
          std::make_shared<FilterOptions>(std::move(*span_opts));
    } else {
      custom_opts = std::make_shared<FilterOptions>(
          FilterOptions::FromExpression(arrow_filter_opts->filter_expression));
    }
    decl->factory_name = "chorys_filter";
    decl->options = std::move(custom_opts);
  }

  if (decl->factory_name == "hash_aggregate" ||
      decl->factory_name == "aggregate") {
    auto agg_opts =
        std::dynamic_pointer_cast<ac::AggregateNodeOptions>(decl->options);
    if (!agg_opts) {
      return arrow::Status::Invalid(
          "Expected AggregateNodeOptions for aggregate declaration");
    }
    if (agg_opts->keys.size() == 1) {
      decl->factory_name = "chorys_group_by";
    }
  }

  if (decl->factory_name == "fetch") {
    auto fetch_opts =
        std::dynamic_pointer_cast<ac::FetchNodeOptions>(decl->options);
    if (!fetch_opts) {
      return arrow::Status::Invalid(
          "Expected FetchNodeOptions for fetch declaration");
    }
    decl->factory_name = "chorys_limit";
  }

  if (decl->factory_name == "hash_join" ||
      decl->factory_name == "hashjoin") {
    auto join_opts =
        std::dynamic_pointer_cast<ac::HashJoinNodeOptions>(decl->options);
    if (!join_opts) {
      return arrow::Status::Invalid(
          "Expected HashJoinNodeOptions for hash_join declaration");
    }
    decl->factory_name = "chorys_hash_join";
  }

  if (decl->factory_name == "order_by") {
    auto order_opts =
        std::dynamic_pointer_cast<ac::OrderByNodeOptions>(decl->options);
    if (!order_opts) {
      return arrow::Status::Invalid(
          "Expected OrderByNodeOptions for order_by declaration");
    }
    decl->factory_name = "chorys_order_by";
  }

  return arrow::Status::OK();
}

}  // namespace chorys::granforge::substrait_bridge
