/**
 * @file GpuSpecialValues_test.cpp
 * @brief NaN/Inf/zero/subnormal propagation on the device.
 *
 * Parameterized over every float-source pair: exact per-format patterns are
 * staged (zero, negative zero, infinities, NaNs, subnormals, min normal,
 * and max finite), converted on the device and with the scalar reference
 * kernel, and compared byte-exact. BF16 subnormal and NaN behavior
 * therefore has direct coverage.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/tensor.h>

#include "utils/GpuCasting.hpp"
#include "utils/Oracle.hpp"

namespace {

using tests::casting::allPairs;
using tests::casting::findPairFor;
using tests::casting::isFloatDtype;
using tests::casting::kDefaultSeed;
using tests::casting::PairInfo;
using tests::casting::readRaw;
using tests::casting::specialValuesOf;
using tests::casting::writeFloatValue;
using tests::casting::writeRaw;
using tests::gpu::castOnDevice;
using tests::gpu::fetchHome;
using tests::gpu::makeDevicePair;
using tests::gpu::sanitizeLabel;
using tests::gpu::scalarRef;
using tests::gpu::stageToDevice;

/// Float-source pairs only.
inline std::vector<PairInfo> floatSourcePairs() {
  std::vector<PairInfo> out;
  for (const auto &p : allPairs()) {
    if (isFloatDtype(p.src)) {
      out.push_back(p);
    }
  }
  return out;
}

/// Parameterized fixture over every float-source cast pair.
class GpuSpecialValues : public ::testing::TestWithParam<PairInfo> {};

} // namespace

/**
 * @brief Verifies special values propagate bitwise on the device.
 */
TEST_P(GpuSpecialValues, SpecialsMatchBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  const PairInfo &prm = GetParam();
  auto probes = specialValuesOf(prm.src);
  if (prm.src == DType_::Float32) {
    probes.push_back({.name = "bf16_min_sub_source", .pattern = 0x00010000U});
    probes.push_back(
        {.name = "bf16_min_sub_source_neg", .pattern = 0x80010000U});
  }
  auto gp = makeDevicePair(prm.src, prm.dst, probes.size(), kDefaultSeed);
  ASSERT_TRUE(gp.ok);
  for (size_t i = 0; i < probes.size(); ++i) {
    writeRaw(gp.host.src.mutableCTensor(), i, probes[i].pattern);
  }
  ASSERT_EQ(stageToDevice(gp).err, novaSuccess) << prm.label;
  ASSERT_EQ(castOnDevice(gp, prm.dst).err, novaSuccess) << prm.label;
  scalarRef(gp, findPairFor(prm.src, prm.dst));
  ASSERT_EQ(fetchHome(gp).err, novaSuccess) << prm.label;
  const Tensor &got = gp.home.getCTensor();
  const Tensor &ref = gp.host.dst.getCTensor();
  for (size_t i = 0U; i < probes.size(); ++i) {
    EXPECT_EQ(readRaw(got, i), readRaw(ref, i))
        << prm.label << " " << probes[i].name;
  }
}

/**
 * @brief Verifies device BF16 NaN outputs carry the canonical payload.
 */
TEST(GpuSpecialValues, Bf16NaNIsCanonical) {
  NOVA_TEST_REQUIRE_GPU();
  const std::array<uint32_t, 4> kNaNPayloads = {0x7FC00000U, 0x7F800001U,
                                                0x7FFFFFFFU, 0xFFC00000U};
  for (const uint32_t bits : kNaNPayloads) {
    SCOPED_TRACE(bits);
    auto gp =
        makeDevicePair(DType_::Float32, DType_::BFloat16, 1, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    writeRaw(gp.host.src.mutableCTensor(), 0, bits);
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, DType_::BFloat16).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::Float32, DType_::BFloat16));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0),
              readRaw(gp.host.dst.getCTensor(), 0));
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), uint64_t{0x7FC0});
  }
  {
    auto gp =
        makeDevicePair(DType_::Float64, DType_::BFloat16, 1, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    writeFloatValue(gp.host.src.mutableCTensor(), 0,
                    std::numeric_limits<double>::quiet_NaN());
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, DType_::BFloat16).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::Float64, DType_::BFloat16));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0),
              readRaw(gp.host.dst.getCTensor(), 0));
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), uint64_t{0x7FC0});
  }
}

INSTANTIATE_TEST_SUITE_P(FloatSources, GpuSpecialValues,
                         ::testing::ValuesIn(floatSourcePairs()),
                         [](const ::testing::TestParamInfo<PairInfo> &info) {
                           return sanitizeLabel(info.param.label);
                         });
