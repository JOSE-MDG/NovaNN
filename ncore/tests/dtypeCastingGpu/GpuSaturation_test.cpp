/**
 * @file GpuSaturation_test.cpp
 * @brief Saturation and integer conversion policies on the device.
 *
 * Targeted probes compare device output against the scalar reference
 * kernel bitwise; absolute expectations pin the bug classes that matter
 * (8-bit saturation vs wrap, NaN to zero, clamp extremes), including
 * the Signed8 to UnSigned8 bit-copy regression pin.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

#include "utils/GpuCasting.hpp"
#include "utils/Oracle.hpp"

namespace {

using tests::casting::findPairFor;
using tests::casting::kDefaultSeed;
using tests::casting::readRaw;
using tests::casting::writeFloatValue;
using tests::casting::writeRaw;
using tests::gpu::castOnDevice;
using tests::gpu::expectHomeEqualRef;
using tests::gpu::fetchHome;
using tests::gpu::makeDevicePair;
using tests::gpu::scalarRef;
using tests::gpu::stageToDevice;

/**
 * @brief Converts staged src values on both paths and compares bytes.
 */
void expectPairBitwise(DType_ src, DType_ dst, size_t n, const char *label) {
  auto gp = makeDevicePair(src, dst, n, kDefaultSeed);
  ASSERT_TRUE(gp.ok);
  ASSERT_EQ(castOnDevice(gp, dst).err, novaSuccess) << label;
  scalarRef(gp, findPairFor(src, dst));
  ASSERT_EQ(fetchHome(gp).err, novaSuccess) << label;
  expectHomeEqualRef(gp, label);
}

} // namespace

/**
 * @brief Verifies float to integer clamps extremes on the device.
 */
TEST(GpuSaturation, FloatToIntClampsToExtremes) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr double kProbes[] = {0.0,   1.0,  -1.0,  127.0, 255.0,
                                256.0, 1e10, -1e10, 1e300, -1e300};
  const DType_ dsts[] = {DType_::Signed8,  DType_::UnSigned8,
                         DType_::Signed16, DType_::UnSigned16,
                         DType_::Signed32, DType_::UnSigned32,
                         DType_::Signed64, DType_::UnSigned64};
  for (const DType_ dst : dsts) {
    auto gp = makeDevicePair(DType_::Float64, dst, 10, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    for (size_t i = 0; i < 10; ++i) {
      writeFloatValue(gp.host.src.mutableCTensor(), i, kProbes[i]);
    }
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, dst).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::Float64, dst));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    expectHomeEqualRef(gp, "f64->int extremes");
  }
  // Absolute anchors: exact small values convert exactly everywhere.
  {
    auto gp =
        makeDevicePair(DType_::Float32, DType_::Signed32, 3, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    Tensor &hs = gp.host.src.mutableCTensor();
    writeFloatValue(hs, 0, 0.0);
    writeFloatValue(hs, 1, 1.0);
    writeFloatValue(hs, 2, -1.0);
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, DType_::Signed32).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::Float32, DType_::Signed32));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    expectHomeEqualRef(gp, "f32 anchors");
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), uint64_t{0});
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 1), uint64_t{1});
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 2), uint64_t{0xFFFFFFFFULL});
  }
}

/**
 * @brief Verifies float NaN maps to zero for integer destinations.
 */
TEST(GpuSaturation, FloatNaNMapsToZero) {
  NOVA_TEST_REQUIRE_GPU();
  const DType_ dsts[] = {DType_::Signed8,  DType_::UnSigned8,
                         DType_::Signed32, DType_::UnSigned32,
                         DType_::Signed64, DType_::UnSigned64};
  for (const DType_ dst : dsts) {
    auto gp = makeDevicePair(DType_::Float32, dst, 2, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    writeFloatValue(
        gp.host.src.mutableCTensor(), 0,
        static_cast<double>(std::numeric_limits<float>::quiet_NaN()));
    writeFloatValue(
        gp.host.src.mutableCTensor(), 1,
        static_cast<double>(-std::numeric_limits<float>::quiet_NaN()));
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, dst).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::Float32, dst));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    expectHomeEqualRef(gp, "f32 nan->int");
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), uint64_t{0});
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 1), uint64_t{0});
  }
}

/**
 * @brief Verifies 8-bit narrowing saturates on the device.
 */
TEST(GpuSaturation, IntNarrowSaturatesInto8Bit) {
  NOVA_TEST_REQUIRE_GPU();
  {
    auto gp =
        makeDevicePair(DType_::Signed16, DType_::Signed8, 4, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    Tensor &hs = gp.host.src.mutableCTensor();
    writeRaw(hs, 0, 0x7FFF);
    writeRaw(hs, 1, 0x8000);
    writeRaw(hs, 2, 200);
    writeRaw(hs, 3, 0xFF38);
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, DType_::Signed8).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::Signed16, DType_::Signed8));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    expectHomeEqualRef(gp, "s16->s8");
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), uint64_t{127});
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 1), uint64_t{0x80});
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 2), uint64_t{127});
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 3), uint64_t{0x80});
  }
  {
    auto gp =
        makeDevicePair(DType_::UnSigned32, DType_::UnSigned8, 2, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    Tensor &hs = gp.host.src.mutableCTensor();
    writeRaw(hs, 0, 300);
    writeRaw(hs, 1, 200);
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, DType_::UnSigned8).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::UnSigned32, DType_::UnSigned8));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    expectHomeEqualRef(gp, "u32->u8");
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), uint64_t{255});
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 1), uint64_t{200});
  }
}

/**
 * @brief Verifies Signed8 to UnSigned8 wraps bit-exact (regression pin).
 */
TEST(GpuSaturation, Signed8ToUnsigned8Wraps) {
  NOVA_TEST_REQUIRE_GPU();
  auto gp = makeDevicePair(DType_::Signed8, DType_::UnSigned8, 3, kDefaultSeed);
  ASSERT_TRUE(gp.ok);
  Tensor &hs = gp.host.src.mutableCTensor();
  writeRaw(hs, 0, 0xFF);
  writeRaw(hs, 1, 0x80);
  writeRaw(hs, 2, 0x7F);
  ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
  ASSERT_EQ(castOnDevice(gp, DType_::UnSigned8).err, novaSuccess);
  scalarRef(gp, findPairFor(DType_::Signed8, DType_::UnSigned8));
  ASSERT_EQ(fetchHome(gp).err, novaSuccess);
  expectHomeEqualRef(gp, "s8->u8 wrap");
  EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), uint64_t{255});
  EXPECT_EQ(readRaw(gp.home.getCTensor(), 1), uint64_t{128});
  EXPECT_EQ(readRaw(gp.home.getCTensor(), 2), uint64_t{127});
}

/**
 * @brief Verifies wider integer narrowing wraps on the device.
 */
TEST(GpuSaturation, IntNarrowWrapsBeyond8Bit) {
  NOVA_TEST_REQUIRE_GPU();
  {
    auto gp =
        makeDevicePair(DType_::Signed32, DType_::Signed16, 2, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    Tensor &hs = gp.host.src.mutableCTensor();
    writeRaw(hs, 0, 0x18000);
    writeRaw(hs, 1, 0x7FFFFFFF);
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, DType_::Signed16).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::Signed32, DType_::Signed16));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    expectHomeEqualRef(gp, "s32->s16 wrap");
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), uint64_t{0x8000});
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 1), uint64_t{0xFFFF});
  }
  {
    auto gp =
        makeDevicePair(DType_::UnSigned32, DType_::UnSigned16, 2, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    Tensor &hs = gp.host.src.mutableCTensor();
    writeRaw(hs, 0, 0x20000);
    writeRaw(hs, 1, 0x12345);
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, DType_::UnSigned16).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::UnSigned32, DType_::UnSigned16));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    expectHomeEqualRef(gp, "u32->u16 wrap");
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), uint64_t{0});
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 1), uint64_t{0x2345});
  }
  {
    auto gp =
        makeDevicePair(DType_::Signed64, DType_::Signed32, 2, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    Tensor &hs = gp.host.src.mutableCTensor();
    writeRaw(hs, 0, 0x100000000ULL);
    writeRaw(hs, 1, 0x1FFFFFFFFULL);
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, DType_::Signed32).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::Signed64, DType_::Signed32));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    expectHomeEqualRef(gp, "s64->s32 wrap");
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), uint64_t{0});
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 1), uint64_t{0xFFFFFFFFULL});
  }
}

/**
 * @brief Verifies integer to float conversion on the device.
 */
TEST(GpuSaturation, IntToFloatMatchesBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  expectPairBitwise(DType_::Signed8, DType_::Float32, 64, "s8->f32");
  expectPairBitwise(DType_::UnSigned64, DType_::Float64, 64, "u64->f64");
  expectPairBitwise(DType_::Signed64, DType_::Float32, 64, "s64->f32");
  expectPairBitwise(DType_::UnSigned32, DType_::Float16, 64, "u32->f16");
}
