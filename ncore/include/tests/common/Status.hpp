/**
 * @file Status.hpp
 * @brief Shared gtest assertion over nova status codes.
 */

#pragma once

#include <gtest/gtest.h>

#include <ncore/core/status.h>

namespace tests::common {

/**
 * @brief Fails the current test unless @p st holds the success code.
 *
 * ASSERT semantics abort the test body as soon as the call fails, so
 * later statements never observe a failed status.
 */
inline void expectOk(const novaStatus_t st, const char *ctx) {
  ASSERT_EQ(st.err, novaSuccess)
      << ctx << (st.message != nullptr ? st.message : "");
}

} // namespace tests::common
