/**
 * \file granforge/bridges/arrow/substrait/extension_registry.hpp
 * \brief Substrait extension registry helpers for DuckDB-compatible plans.
 */
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <arrow/engine/substrait/extension_set.h>
#include <arrow/engine/substrait/util.h>

namespace chorys::granforge::substrait_bridge {

namespace eng = ::arrow::engine;

/// \brief Build a Substrait extension registry that maps DuckDB calls to Arrow kernels.
inline std::shared_ptr<eng::ExtensionIdRegistry> GetSubstraitExtensionRegistry() {
  static std::shared_ptr<eng::ExtensionIdRegistry> registry = [] {
    auto reg = eng::nested_extension_id_registry(eng::default_extension_id_registry());

    auto add_mapping = [&reg](const std::string& name, const std::string& arrow_func) {
      eng::Id id{
          "https://github.com/substrait-io/substrait/blob/main/extensions/",
          name};
      auto st = reg->AddSubstraitCallToArrow(id, arrow_func);
      if (!st.ok()) {
        throw std::runtime_error(
            "Failed to add Substrait mapping for " + name + ": " + st.ToString());
      }
    };

    add_mapping("lt", "less");
    add_mapping("lt:i32_i32", "less");
    add_mapping("gt", "greater");
    add_mapping("gt:i32_i32", "greater");
    add_mapping("gte", "greater_equal");
    add_mapping("gte:i32_i32", "greater_equal");
    add_mapping("eq", "equal");
    add_mapping("eq:i32_i32", "equal");
    add_mapping("equal", "equal");
    add_mapping("equal:i32_i32", "equal");
    add_mapping("neq", "not_equal");
    add_mapping("neq:i32_i32", "not_equal");
    add_mapping("not_equal", "not_equal");
    add_mapping("not_equal:i32_i32", "not_equal");

    return reg;
  }();

  return registry;
}

}  // namespace chorys::granforge::substrait_bridge
