/**
 * \file granforge/bridges/arrow/substrait/plan_io.hpp
 * \brief Helpers for reading Substrait plans from stdin.
 */
#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace chorys::granforge::substrait_bridge {

/// \brief Slurp stdin into a string; Substrait plans are handed over as binary blobs.
inline std::string ReadAllStdin() {
  std::ostringstream buf;
  buf << std::cin.rdbuf();
  return buf.str();
}

}  // namespace chorys::granforge::substrait_bridge
