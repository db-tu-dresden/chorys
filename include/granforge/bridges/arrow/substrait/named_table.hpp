/**
 * \file granforge/bridges/arrow/substrait/named_table.hpp
 * \brief Helpers for mapping Substrait named tables to custom scan declarations.
 */
#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arrow/acero/exec_plan.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include "granforge/bridges/arrow/ops/read.hpp"

namespace chorys::granforge::substrait_bridge {

namespace ac = ::arrow::acero;

/// \brief Construct a declaration that scans a single Parquet file.
inline arrow::Result<ac::Declaration> MakeParquetScanDeclaration(
    const std::string& parquet_path) {
  chorys::granforge::arrow_bridge::ParquetScanOptions scan_opts;
  scan_opts.path = parquet_path;
  return ac::Declaration{"chorys_pq_scan", std::move(scan_opts)};
}

/// \brief Parse a TABLE=PATH argument from the CLI.
inline arrow::Result<std::pair<std::string, std::string>> ParseTablePathArg(
    std::string_view kv) {
  auto pos = kv.find('=');
  if (pos == std::string_view::npos || pos == 0 || pos == kv.size() - 1) {
    return arrow::Status::Invalid(
        "Expected --data to be in TABLE=PATH form, got '", kv, "'");
  }
  return std::make_pair(std::string(kv.substr(0, pos)),
                        std::string(kv.substr(pos + 1)));
}

/// \brief Build a NamedTableProvider from a simple name->path map.
inline auto MakeNamedTableProvider(
    std::unordered_map<std::string, std::string> table_paths) {
  return [table_paths = std::move(table_paths)](
             const std::vector<std::string>& names,
             const arrow::Schema&) -> arrow::Result<ac::Declaration> {
    if (names.empty()) {
      return arrow::Status::Invalid("NamedTableProvider: empty table name");
    }
    if (names.size() > 1) {
      return arrow::Status::Invalid(
          "NamedTableProvider: multi-part table names not supported");
    }

    const std::string& table = names[0];
    const auto it = table_paths.find(table);
    if (it != table_paths.end()) {
      return MakeParquetScanDeclaration(it->second);
    }

    return arrow::Status::Invalid(
        "NamedTableProvider: unknown table '" + table + "'");
  };
}

}  // namespace chorys::granforge::substrait_bridge
