/**
 * @file GpuOracleEquivalence_test.cpp
 * @brief Full-matrix bitwise equivalence of device casts vs CPU scalar.
 *
 * Parameterized over all 210 supported pairs: each case stages the
 * deterministic fill to the device, converts on the device and with
 * the scalar reference kernel, transfers the device result home, and
 * compares byte-exact (strict bitwise contract, NaN payloads
 * included). The scalar kernel is the pinned portable semantic both
 * backends mirror.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>

#include "utils/GpuCasting.hpp"
#include "utils/Oracle.hpp"

namespace {

using tests::casting::allPairs;
using tests::casting::kDefaultSeed;
using tests::casting::PairInfo;
using tests::gpu::castOnDevice;
using tests::gpu::expectHomeEqualRef;
using tests::gpu::fetchHome;
using tests::gpu::makeDevicePair;
using tests::gpu::sanitizeLabel;
using tests::gpu::scalarRef;

/// Parameterized fixture over every supported cast pair.
class GpuOracleEquivalence : public ::testing::TestWithParam<PairInfo> {};

} // namespace

/**
 * @brief Verifies one pair converts bitwise-identically on the device.
 */
TEST_P(GpuOracleEquivalence, DeviceMatchesCpuBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  const PairInfo &prm = GetParam();
  constexpr size_t kElems = 64;
  auto gp = makeDevicePair(prm.src, prm.dst, kElems, kDefaultSeed);
  ASSERT_TRUE(gp.ok);
  ASSERT_EQ(castOnDevice(gp, prm.dst).err, novaSuccess) << prm.label;
  scalarRef(gp, &prm);
  ASSERT_EQ(fetchHome(gp).err, novaSuccess) << prm.label;
  expectHomeEqualRef(gp, prm.label.c_str());
}

INSTANTIATE_TEST_SUITE_P(AllPairs, GpuOracleEquivalence,
                         ::testing::ValuesIn(allPairs()),
                         [](const ::testing::TestParamInfo<PairInfo> &info) {
                           return sanitizeLabel(info.param.label);
                         });
