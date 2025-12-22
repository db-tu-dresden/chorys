/**
 * \file granforge/bridges/arrow/ops/read.hpp
 * \brief Source ExecNode that reads Parquet files using the dbops scan functor.
 *
 * This node bridges the `dbops::ParquetScanFunctor` with Arrow Acero’s execution graph,
 * allowing Substrait named tables to be bound to physical Parquet files while keeping the
 * rest of the pipeline unaware of Arrow’s dataset layer.
 */
#pragma once

#include <memory>
#include <utility>

#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>
#include <arrow/compute/exec.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include "granforge/bridges/arrow/ops/parquet_scan_functor.hpp"

namespace chorys::granforge::arrow_bridge {

namespace ac = ::arrow::acero;
namespace cmpt = ::arrow::compute;
namespace db = chorys::granforge::dbops;

/// \brief Options object containing the Parquet file path for the scan node.
struct ParquetScanOptions : public ac::ExecNodeOptions {
  std::string path;
};

/// \brief Arrow ExecNode that emits batches read from a Parquet file.
class ParquetScanNode : public ac::ExecNode {
 public:
  static arrow::Result<ac::ExecNode*> Make(ac::ExecPlan* plan,
                                           std::vector<ac::ExecNode*> inputs,
                                           const ac::ExecNodeOptions& options) {
    if (!inputs.empty()) {
      return arrow::Status::Invalid(
          "ParquetScanNode is a source and must have no inputs");
    }

    auto& opts = static_cast<const ParquetScanOptions&>(options);

    db::ParquetScanConfig config{opts.path};
    ARROW_ASSIGN_OR_RAISE(auto functor,
                          ParquetScanFunctor::Make(std::move(config)));
    auto schema = functor->schema();

    auto node = std::unique_ptr<ParquetScanNode>(
        new ParquetScanNode(plan, std::move(schema), std::move(functor)));
    auto* out = plan->AddNode(std::move(node));
    return out;
  }

  /// \brief ExecNode descriptor for explain output.
  const char* kind_name() const override { return "chorys_pq_scan"; }

  /// Parquet scan nodes are sources, so receiving input is an error.
  arrow::Status InputReceived(ac::ExecNode*, cmpt::ExecBatch batch) override {
    ARROW_UNUSED(batch);
    return arrow::Status::Invalid(
        "ParquetScanNode is a source and should not receive input");
  }

  arrow::Status InputFinished(ac::ExecNode*, int) override {
    return arrow::Status::OK();
  }

  void PauseProducing(ac::ExecNode*, int32_t) override {}

  void ResumeProducing(ac::ExecNode*, int32_t) override {}

  /// Pull record batches from the functor and stream them downstream.
  arrow::Status StartProducing() override {
    int produced_batches = 0;
    while (true) {
      ARROW_ASSIGN_OR_RAISE(auto maybe_batch, functor_->NextBatch());
      if (!maybe_batch.has_value()) {
        break;
      }
      ++produced_batches;
      ARROW_RETURN_NOT_OK(
          this->output_->InputReceived(this, std::move(*maybe_batch)));
    }

    ARROW_RETURN_NOT_OK(
        this->output_->InputFinished(this, produced_batches));
    return arrow::Status::OK();
  }

  arrow::Status StopProducing() override { return StopProducingImpl(); }

 protected:
  arrow::Status StopProducingImpl() override { return arrow::Status::OK(); }

 private:
  ParquetScanNode(ac::ExecPlan* plan,
                  std::shared_ptr<arrow::Schema> output_schema,
                  std::unique_ptr<ParquetScanFunctor> functor)
      : ac::ExecNode(plan,
                     /*inputs=*/{},
                     /*input_labels=*/{},
                     std::move(output_schema)),
        functor_(std::move(functor)) {}

  std::unique_ptr<ParquetScanFunctor> functor_;
};

}  // namespace chorys::granforge::arrow_bridge
