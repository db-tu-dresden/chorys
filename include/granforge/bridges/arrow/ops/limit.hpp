/**
 * \file granforge/bridges/arrow/ops/limit.hpp
 * \brief ExecNode that mirrors Substrait/DuckDB LIMIT/OFFSET semantics using dbops trackers.
 *
 * Arrow Acero exposes a FetchNode, but this bridge allows us to intercept those declarations and
 * monitor how many rows were produced without relying on Arrow's default implementation.  It is
 * intentionally straightforward: batches are sliced in place while the tracker keeps the global
 * offset/count bookkeeping across multiple upstream batches.
 */
#pragma once

#include <memory>
#include <utility>

#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>
#include <arrow/compute/exec.h>
#include <arrow/datum.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include "granforge/dbops/limit_functor.hpp"

namespace chorys::granforge::arrow_bridge {

namespace ac = ::arrow::acero;
namespace cp = ::arrow::compute;
namespace db = chorys::granforge::dbops;

/// \brief Minimal ExecNode that enforces LIMIT/OFFSET using dbops::LimitTracker.
class LimitNode : public ac::ExecNode {
 public:
  /// \brief Instantiate the LIMIT node from Fetch options.
  static arrow::Result<ac::ExecNode*> Make(ac::ExecPlan* plan,
                                           std::vector<ac::ExecNode*> inputs,
                                           const ac::ExecNodeOptions& options) {
    if (inputs.size() != 1) {
      return arrow::Status::Invalid("LimitNode expects exactly one input");
    }

    const auto& fetch_opts = static_cast<const ac::FetchNodeOptions&>(options);
    if (fetch_opts.count < 0) {
      return arrow::Status::Invalid("LimitNode requires non-negative count");
    }

    auto schema = inputs[0]->output_schema();
    auto node = std::unique_ptr<LimitNode>(
        new LimitNode(plan, std::move(inputs), std::move(schema),
                      fetch_opts.offset, fetch_opts.count));
    return plan->AddNode(std::move(node));
  }

  /// \brief ExecNode descriptor for explain output.
  const char* kind_name() const override { return "chorys_limit"; }

  /// Inspect the incoming batch, slice out the relevant portion, and pass it downstream.
  arrow::Status InputReceived(ac::ExecNode*, cp::ExecBatch batch) override {
    if (limit_tracker_.Done()) {
      return arrow::Status::OK();
    }

    int64_t start = 0;
    int64_t length = 0;
    if (!limit_tracker_.SelectRange(batch.length, &start, &length)) {
      return arrow::Status::OK();
    }

    cp::ExecBatch sliced = batch.Slice(start, length);
    return this->output_->InputReceived(this, std::move(sliced));
  }

  /// Propagate InputFinished downstream once the parent nodes finish producing.
  /// \brief Propagate InputFinished downstream once upstream is done.
  arrow::Status InputFinished(ac::ExecNode*, int32_t) override {
    return this->output_->InputFinished(this, 1);
  }

  void PauseProducing(ac::ExecNode*, int32_t) override {}
  void ResumeProducing(ac::ExecNode*, int32_t) override {}

  /// \brief Begin streaming slices downstream.
  arrow::Status StartProducing() override { return arrow::Status::OK(); }
  /// \brief Stop production and release per-node resources.
  arrow::Status StopProducing() override { return StopProducingImpl(); }

 protected:
  /// \brief Concrete implementation used by StopProducing.
  arrow::Status StopProducingImpl() override { return arrow::Status::OK(); }

 private:
  LimitNode(ac::ExecPlan* plan, std::vector<ac::ExecNode*> inputs,
            std::shared_ptr<arrow::Schema> output_schema, int64_t offset,
            int64_t count)
      : ac::ExecNode(plan, std::move(inputs), std::vector<std::string>{"input"},
                     std::move(output_schema)),
        limit_tracker_(offset, count) {}

  db::LimitTracker limit_tracker_;
};

}  // namespace chorys::granforge::arrow_bridge
