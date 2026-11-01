/**
 * @file Naming.hpp
 * @brief Label sanitization for gtest value-parameterized test names.
 */

#pragma once

#include <string>

namespace tests::common {

/**
 * @brief Replaces the arrow glyphs of pair labels with gtest-safe text.
 *
 * gtest embeds the printed parameter in the TEST_P identifier, and a
 * '-' or '>' there is a compile error, so "fp8->fp32" becomes
 * "fp8_fp32".
 */
inline std::string sanitize(std::string label) {
  for (char &c : label) {
    if (c == '-' || c == '>') {
      c = '_';
    }
  }
  return label;
}

} // namespace tests::common
