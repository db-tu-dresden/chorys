/**
 * \file src/examples/granforge_demo.cpp
 * \brief Command-line demo backend that executes rewritten Substrait plans via Arrow/Acero.
 *
 * DuckDB produces Substrait plans that this binary consumes on stdin.  Specific nodes are
 * rewritten to use the dbops implementations (Parquet scan, filter, group-by, order-by, limit),
 * executed inside an Arrow ExecPlan, and the results are serialized to JSON for comparison with
 * DuckDB's direct execution path.
 */

#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <memory>
#include <unordered_map>
#include <utility>
#include <cstdlib>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <arrow/buffer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/record_batch.h>
#include <arrow/scalar.h>

#include <arrow/compute/api.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>

#include <arrow/engine/substrait/api.h>
#include <arrow/engine/substrait/options.h>
#include <arrow/engine/substrait/serde.h>
#include <arrow/engine/substrait/util.h>

#include <nlohmann/json.hpp>

#include "granforge/bridges/arrow/ops/register_ops.hpp"
#include "granforge/bridges/arrow/substrait/extension_registry.hpp"
#include "granforge/bridges/arrow/substrait/named_table.hpp"
#include "granforge/bridges/arrow/substrait/plan_io.hpp"
#include "granforge/bridges/arrow/substrait/plan_rewrite.hpp"

namespace eng = arrow::engine;
namespace ac  = arrow::acero;
namespace cp  = arrow::compute;
namespace sb  = chorys::granforge::substrait_bridge;

namespace {

/// \brief Initialize spdlog so dbops components can emit trace/debug logs.
void InitializeFileLogger() {
  const char* path_env = std::getenv("GRANFORGE_LOG_PATH");
  std::string log_path = (path_env && *path_env) ? path_env : "granforge_backend.log";
  try {
    auto file_logger =
        spdlog::basic_logger_mt("granforge_demo", log_path, true);
    spdlog::set_default_logger(file_logger);
    spdlog::set_level(spdlog::level::trace);
    spdlog::flush_on(spdlog::level::info);
  } catch (const spdlog::spdlog_ex& ex) {
    throw std::runtime_error("Failed to initialize logger at " + log_path + ": " +
                             std::string(ex.what()));
  }
}

}  // namespace

/// \brief Convert an Arrow scalar into a JSON value (used when comparing outputs).
static nlohmann::json ScalarToJson(const arrow::Scalar& scalar) {
  if (!scalar.is_valid) {
    return nullptr;
  }
  switch (scalar.type->id()) {
    case arrow::Type::INT64:
      return static_cast<const arrow::Int64Scalar&>(scalar).value;
    case arrow::Type::INT32:
      return static_cast<const arrow::Int32Scalar&>(scalar).value;
    case arrow::Type::DOUBLE:
      return static_cast<const arrow::DoubleScalar&>(scalar).value;
    case arrow::Type::FLOAT:
      return static_cast<const arrow::FloatScalar&>(scalar).value;
    case arrow::Type::BOOL:
      return static_cast<const arrow::BooleanScalar&>(scalar).value;
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
      return static_cast<const arrow::BaseBinaryScalar&>(scalar).value->ToString();
    default:
      return scalar.ToString();
  }
}

/// \brief Serialize an Arrow table row-by-row into JSON for the CLI output.
static arrow::Result<nlohmann::json> TableToJson(
    const std::shared_ptr<arrow::Table>& table) {
  nlohmann::json rows = nlohmann::json::array();
  auto schema = table->schema();
  for (int64_t r = 0; r < table->num_rows(); ++r) {
    nlohmann::json row = nlohmann::json::object();
    for (int c = 0; c < schema->num_fields(); ++c) {
      ARROW_ASSIGN_OR_RAISE(auto scalar, table->column(c)->GetScalar(r));
      row[schema->field(c)->name()] = ScalarToJson(*scalar);
    }
    rows.push_back(std::move(row));
  }
  return rows;
}


// ---------------------------------------------------------------------
// Helper: usage banner

static void PrintUsage(const char* prog) {
  std::cout << "Usage: " << prog << " [--data TABLE=PATH]...\n"
            << "\n"
            << "Reads a binary Substrait plan from stdin, resolves each named\n"
            << "table via the provided --data mappings, executes it with Apache\n"
            << "Arrow, and prints the resulting Arrow table.\n"
            << "\n"
            << "Options:\n"
            << "  --data TABLE=PATH   Bind Substrait table name TABLE to a Parquet file PATH.\n"
            << "                      Repeat for multiple tables. At least one mapping is\n"
            << "                      required unless COMPLETE_CAST_PARQUET is set.\n"
            << "  -h, --help          Show this help and exit.\n";
}

// ---------------------------------------------------------------------
// main:
// 1. Initialize Arrow compute
// 2. Create ExecContext (memory pool + thread pool)
// 3. Read binary Substrait plan from stdin
// 4. Configure NamedTableProvider for "complete_cast"
// 5. Execute Substrait plan via ExecuteSerializedPlan
// 6. Collect to Table and print

/// \brief Program entry point – rewrites, registers, executes, and prints Substrait plans.
int main(int argc, char** argv) {
  InitializeFileLogger();
  std::atexit([] { spdlog::shutdown(); });
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
  }

  // Initialize compute
  arrow::Status st = cp::Initialize();
  if (!st.ok()) {
    std::cerr << "Failed to initialize Arrow compute: "
              << st.ToString() << "\n";
    return 1;
  }

  // 1) Memory pool (default is fine unless you need custom tracking)
  arrow::MemoryPool* pool = arrow::default_memory_pool();

  // 4) Read binary Substrait plan from stdin
  std::string bin_plan = sb::ReadAllStdin();
  if (bin_plan.empty()) {
    std::cerr << "No Substrait binary plan received on stdin.\n";
    return 1;
  }

  // Wrap it in an Arrow Buffer (view over bin_plan's memory)
  arrow::Buffer buf(
      reinterpret_cast<const uint8_t*>(bin_plan.data()),
      static_cast<int64_t>(bin_plan.size()));

  // 5) Execute Substrait plan via Acero and collect results

  // Our custom registry (with lt / lt:i32_i32 -> "less")
  auto ext_registry = sb::GetSubstraitExtensionRegistry();

  // Register the custom nodes used by the NamedTableProvider and filter rewrites
  auto ensure_registered = [](const arrow::Status& status,
                              const char* name) -> bool {
    if (status.ok() || status.IsAlreadyExists()) {
      return true;
    }
    std::cerr << "Failed to register " << name << ": "
              << status.ToString() << "\n";
    return false;
  };

  if (!ensure_registered(chorys::granforge::arrow_bridge::RegisterParquetScanNode(),
                         "chorys_pq_scan")) {
    return 1;
  }
  if (!ensure_registered(chorys::granforge::arrow_bridge::RegisterFilterNode(),
                         "chorys_filter")) {
    return 1;
  }
  if (!ensure_registered(chorys::granforge::arrow_bridge::RegisterGroupByNode(),
                         "chorys_group_by")) {
    return 1;
  }
  if (!ensure_registered(chorys::granforge::arrow_bridge::RegisterOrderByNode(),
                         "chorys_order_by")) {
    return 1;
  }
  if (!ensure_registered(chorys::granforge::arrow_bridge::RegisterLimitNode(),
                         "chorys_limit")) {
    return 1;
  }
  if (!ensure_registered(chorys::granforge::arrow_bridge::RegisterHashJoinNode(),
                         "chorys_hash_join")) {
    return 1;
  }

  // Arrow compute function registry (for "less", aggregates, etc.)
  cp::FunctionRegistry* func_registry = cp::GetFunctionRegistry();

  // Conversion options (we'll configure named_table_provider here)
  eng::ConversionOptions conv_opts;
  
  std::unordered_map<std::string, std::string> table_paths;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    std::string value;
    if (arg.rfind("--data=", 0) == 0) {
      value = arg.substr(std::string("--data=").size());
    } else if (arg == "--data") {
      if (i + 1 >= argc) {
        std::cerr << "--data requires TABLE=PATH argument\n";
        return 1;
      }
      value = argv[++i];
    } else {
      continue;
    }

    auto parsed = sb::ParseTablePathArg(value);
    if (!parsed.ok()) {
      std::cerr << parsed.status().ToString() << "\n";
      return 1;
    }
    auto [table_name, table_path] = *parsed;
    table_paths[table_name] = table_path;
  }

  if (table_paths.empty()) {
    std::cerr << "No table mappings provided. Use --data table=path or set COMPLETE_CAST_PARQUET.\n";
    return 1;
  }

  // NamedTableProvider: maps Substrait named tables -> Acero Declarations
  conv_opts.named_table_provider = sb::MakeNamedTableProvider(table_paths);

  // Deserialize Substrait plan into an Acero declaration
  arrow::Result<eng::PlanInfo> info_result = eng::DeserializePlan(
      buf,
      /*registry=*/ext_registry.get(),
      /*ext_set_out=*/nullptr,
      /*conversion_options=*/conv_opts);

  if (!info_result.ok()) {
    std::cerr << "Failed to deserialize Substrait plan: "
              << info_result.status().ToString() << "\n";
    return 1;
  }

  eng::PlanInfo plan_info = *info_result;

  auto original_decl_str =
      ac::DeclarationToString(plan_info.root.declaration, func_registry);
  if (original_decl_str.ok()) {
    spdlog::info("Original plan:\n{}", *original_decl_str);
  }

  // Replace selected nodes with our custom implementations
  arrow::Status rewrite_st = sb::RewritePlan(&plan_info.root.declaration);
  if (!rewrite_st.ok()) {
    std::cerr << "Failed to rewrite declarations: "
              << rewrite_st.ToString() << "\n";
    return 1;
  }

  auto decl_str = ac::DeclarationToString(plan_info.root.declaration, func_registry);
  if (decl_str.ok()) {
    spdlog::info("Execution plan:\n{}", *decl_str);
  } else {
    spdlog::warn("Failed to stringify declaration: {}", decl_str.status().ToString());
  }

  // Execute declaration and collect to table
  ac::QueryOptions query_opts;
  query_opts.use_threads = true;
  query_opts.memory_pool = pool;
  query_opts.function_registry = func_registry;
  if (!plan_info.names.empty()) {
    query_opts.field_names = plan_info.names;
  }

  arrow::Result<std::shared_ptr<arrow::Table>> table_result =
      ac::DeclarationToTable(std::move(plan_info.root.declaration), query_opts);

  if (!table_result.ok()) {
    std::cerr << "Failed to execute declaration: "
              << table_result.status().ToString() << "\n";
    return 1;
  }
  
  std::shared_ptr<arrow::Table> table = *table_result;

  // Print result
  std::cout << "Result table:\n";
  std::cout << table->ToString() << "\n";

  auto json_result = TableToJson(table);
  if (!json_result.ok()) {
    std::cerr << "Failed to serialize result to JSON: "
              << json_result.status().ToString() << "\n";
    return 1;
  }
  std::cout << "JSON_RESULT:" << json_result->dump() << "\n";

  return 0;
}
