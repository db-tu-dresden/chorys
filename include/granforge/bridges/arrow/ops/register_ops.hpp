/**
 * \file granforge/bridges/arrow/ops/register_ops.hpp
 * \brief Helpers that register all custom ExecNode factories with Arrow Acero.
 *
 * Arrow’s ExecFactoryRegistry requires registration to be explicit and guarded.  These helpers
 * centralize the registration logic so the backend can ensure every custom node is present before
 * executing a deserialized plan.
 */
#pragma once
#include <arrow/acero/exec_plan.h>

#include "granforge/bridges/arrow/ops/filter.hpp"
#include "granforge/bridges/arrow/ops/group_by.hpp"
#include "granforge/bridges/arrow/ops/hash_join.hpp"
#include "granforge/bridges/arrow/ops/order_by.hpp"
#include "granforge/bridges/arrow/ops/limit.hpp"
#include "granforge/bridges/arrow/ops/read.hpp"

namespace chorys::granforge::arrow_bridge {
namespace ac = ::arrow::acero;

/// \brief Register the Parquet scan node that exposes dbops-aware Parquet reading.
inline arrow::Status RegisterParquetScanNode() {
  ac::ExecFactoryRegistry* registry = ac::default_exec_factory_registry();
  return registry->AddFactory("chorys_pq_scan", &ParquetScanNode::Make);
}

/// \brief Register the filter node that forwards to custom filter functors.
inline arrow::Status RegisterFilterNode() {
  ac::ExecFactoryRegistry* registry = ac::default_exec_factory_registry();
  return registry->AddFactory("chorys_filter", &FilterNode::Make);
}

/// \brief Register the lightweight group-by implementation.
inline arrow::Status RegisterGroupByNode() {
  ac::ExecFactoryRegistry* registry = ac::default_exec_factory_registry();
  return registry->AddFactory("chorys_group_by", &GroupByNode::Make);
}

/// \brief Register the ORDER BY implementation backed by dbops.
inline arrow::Status RegisterOrderByNode() {
  ac::ExecFactoryRegistry* registry = ac::default_exec_factory_registry();
  return registry->AddFactory("chorys_order_by", &OrderByNode::Make);
}

/// \brief Register the LIMIT/OFFSET implementation backed by dbops.
inline arrow::Status RegisterLimitNode() {
  ac::ExecFactoryRegistry* registry = ac::default_exec_factory_registry();
  return registry->AddFactory("chorys_limit", &LimitNode::Make);
}

/// \brief Register the hash join node that leverages dbops hash tables.
inline arrow::Status RegisterHashJoinNode() {
  ac::ExecFactoryRegistry* registry = ac::default_exec_factory_registry();
  return registry->AddFactory("chorys_hash_join", &HashJoinNode::Make);
}

}  // namespace chorys::granforge::arrow_bridge
