/**
 * @file GpuDispatch_test.cpp
 * @brief Device dispatch coverage for the GPU dtype casting path.
 *
 * Verifies the 210-pair registry resolves in GPU test binaries,
 * converts absolute-value smoke cases on the active backend, covers
 * a large multi-block tensor, and pins the status edges that need no
 * device launch (shape mismatch and empty tensors through the
 * backend launcher directly; unsupported and identity pairs through
 * cast() on CPU tensors).
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

#include "utils/GpuCasting.hpp"
#include "utils/Oracle.hpp"

namespace {

using ncore::wrappers::TensorCXX;
using tests::casting::allPairs;
using tests::casting::findPairFor;
using tests::casting::kDefaultSeed;
using tests::casting::readRaw;
using tests::casting::writeRaw;
using tests::gpu::callBackendLauncher;
using tests::gpu::expectHomeEqualRef;
using tests::gpu::makeDevicePair;
using tests::gpu::scalarRef;

/// One absolute conversion with exact expectations in both formats.
struct GpuFamilySample {
  DType_ src;
  DType_ dst;
  uint64_t srcPattern;
  uint64_t dstPattern;
  const char *label;
};

/// Representatives mirroring the CPU dispatch families.
constexpr std::array<GpuFamilySample, 7> kGpuFamilySamples{{
    {.src = DType_::Float16,
     .dst = DType_::Float32,
     .srcPattern = 0x3C00,
     .dstPattern = 0x3F800000,
     .label = "fp->fp"},
    {.src = DType_::Float32,
     .dst = DType_::Signed8,
     .srcPattern = 0x40600000,
     .dstPattern = 0x03,
     .label = "fp->int"},
    {.src = DType_::Signed8,
     .dst = DType_::Float32,
     .srcPattern = 0xFD,
     .dstPattern = 0xC0400000,
     .label = "int->fp"},
    {.src = DType_::Signed8,
     .dst = DType_::Signed32,
     .srcPattern = 0xFB,
     .dstPattern = 0xFFFFFFFBULL,
     .label = "s->s"},
    {.src = DType_::UnSigned8,
     .dst = DType_::UnSigned32,
     .srcPattern = 0xC8,
     .dstPattern = 0xC8,
     .label = "u->u"},
    {.src = DType_::Signed16,
     .dst = DType_::UnSigned8,
     .srcPattern = 0xFFFF,
     .dstPattern = 0x00,
     .label = "s->u"},
    {.src = DType_::UnSigned64,
     .dst = DType_::Signed8,
     .srcPattern = 100,
     .dstPattern = 100,
     .label = "u->s"},
}};

} // namespace

/**
 * @brief Verifies the 210-pair registry resolves in GPU test binaries.
 */
TEST(GpuDispatch, AllSupportedPairsResolve) {
  const auto &pairs = allPairs();
  ASSERT_EQ(pairs.size(), size_t{210});
  for (const auto &p : pairs) {
    EXPECT_NE(get_dispatched_cast_func(p.src, p.dst), nullptr) << p.label;
  }
}

/**
 * @brief Converts one absolute value per family on the device.
 */
TEST(GpuDispatch, FamilySmokeValuesMatchBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  for (const auto &sample : kGpuFamilySamples) {
    SCOPED_TRACE(sample.label);
    auto gp = makeDevicePair(sample.src, sample.dst, 4, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    writeRaw(gp.host.src.mutableCTensor(), 0, sample.srcPattern);
    ASSERT_EQ(stageToDevice(gp).err, novaSuccess);
    ASSERT_EQ(castOnDevice(gp, sample.dst).err, novaSuccess);
    scalarRef(gp, findPairFor(sample.src, sample.dst));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    expectHomeEqualRef(gp, sample.label);
    EXPECT_EQ(readRaw(gp.home.getCTensor(), 0), sample.dstPattern)
        << sample.label;
  }
}

/**
 * @brief Converts a single-element tensor on the device.
 */
TEST(GpuDispatch, SizeOneMatchesBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  auto gp = makeDevicePair(DType_::Signed32, DType_::Float32, 1, kDefaultSeed);
  ASSERT_TRUE(gp.ok);
  ASSERT_EQ(castOnDevice(gp, DType_::Float32).err, novaSuccess);
  scalarRef(gp, findPairFor(DType_::Signed32, DType_::Float32));
  ASSERT_EQ(fetchHome(gp).err, novaSuccess);
  expectHomeEqualRef(gp, "s32->f32 size 1");
}

/**
 * @brief Converts a multi-block tensor on the device.
 */
TEST(GpuDispatch, LargeTensorMatchesBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kElems = size_t{1} << 20;
  auto gp =
      makeDevicePair(DType_::Float32, DType_::Float64, kElems, kDefaultSeed);
  ASSERT_TRUE(gp.ok);
  ASSERT_EQ(castOnDevice(gp, DType_::Float64).err, novaSuccess);
  scalarRef(gp, findPairFor(DType_::Float32, DType_::Float64));
  ASSERT_EQ(fetchHome(gp).err, novaSuccess);
  expectHomeEqualRef(gp, "f32->f64 1M");
}

/**
 * @brief Converts sizes that exercise masked coarsened-loop tails.
 */
TEST(GpuDispatch, TailSizesMatchBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr std::array<size_t, 4> kTailSizes{{3, 5, 65, 257}};
  for (const size_t elems : kTailSizes) {
    SCOPED_TRACE(elems);
    auto gp =
        makeDevicePair(DType_::Float32, DType_::Float64, elems, kDefaultSeed);
    ASSERT_TRUE(gp.ok);
    ASSERT_EQ(castOnDevice(gp, DType_::Float64).err, novaSuccess);
    scalarRef(gp, findPairFor(DType_::Float32, DType_::Float64));
    ASSERT_EQ(fetchHome(gp).err, novaSuccess);
    expectHomeEqualRef(gp, "f32->f64 tail");
  }
}

/**
 * @brief Verifies packed FP4 device pairs relate widths by two.
 */
TEST(GpuDispatch, Fp4DevicePairWidthsMatch) {
  NOVA_TEST_REQUIRE_GPU();
  auto up =
      makeDevicePair(DType_::Float4E2M1fn, DType_::Float32, 10, kDefaultSeed);
  ASSERT_TRUE(up.ok);
  EXPECT_EQ(up.devDst.getSize(), up.devSrc.getSize() * 2U);
  auto pk =
      makeDevicePair(DType_::Float32, DType_::Float4E2M1fn, 10, kDefaultSeed);
  ASSERT_TRUE(pk.ok);
  EXPECT_EQ(pk.devSrc.getSize(), pk.devDst.getSize() * 2U);
  ASSERT_EQ(castOnDevice(up, DType_::Float32).err, novaSuccess);
  scalarRef(up, findPairFor(DType_::Float4E2M1fn, DType_::Float32));
  ASSERT_EQ(fetchHome(up).err, novaSuccess);
  expectHomeEqualRef(up, "fp4 unpack");
  ASSERT_EQ(castOnDevice(pk, DType_::Float4E2M1fn).err, novaSuccess);
  scalarRef(pk, findPairFor(DType_::Float32, DType_::Float4E2M1fn));
  ASSERT_EQ(fetchHome(pk).err, novaSuccess);
  expectHomeEqualRef(pk, "fp4 pack");
}

/**
 * @brief Verifies shape mismatch reports without any device launch.
 *
 * Uses hand-built metadata structs because the shape check precedes
 * the device query; no hardware is required.
 */
TEST(GpuDispatch, ShapeMismatchReportsStatus) {
#if !defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
  GTEST_SKIP() << "no GPU backend compiled in";
#else
  Tensor src{};
  src.size = 8;
  src.dtype = DType_::Float32;
  Tensor dst{};
  dst.size = 4;
  dst.dtype = DType_::Float16;
  EXPECT_EQ(callBackendLauncher(&src, &dst).err, novaShapeMismatch);
#endif
}

/**
 * @brief Verifies empty tensors succeed without any device launch.
 *
 * Uses hand-built metadata structs because the public tensor API
 * cannot construct zero-size tensors; the launcher returns before
 * any device query, so no hardware is required.
 */
TEST(GpuDispatch, EmptyTensorsSucceed) {
#if !defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
  GTEST_SKIP() << "no GPU backend compiled in";
#else
  Tensor src{};
  src.size = 0;
  src.dtype = DType_::Float32;
  Tensor dst{};
  dst.size = 0;
  dst.dtype = DType_::Float16;
  EXPECT_EQ(callBackendLauncher(&src, &dst).err, novaSuccess);
#endif
}

/**
 * @brief Verifies identity and quantized pairs report unsupported.
 *
 * Runs on CPU tensors through cast(), so no hardware is required:
 * both cases land in the null-dispatch guard.
 */
TEST(GpuDispatch, UnsupportedPairsReportStatus) {
  novaStatus_t st{};
  TensorCXX src({4}, DType_::Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX dst({4}, DType_::Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.mutableCTensor();
  Tensor dv = dst.mutableCTensor();
  EXPECT_EQ(cast(&sv, &dv, DType_::Float32).err, novaCastNotSupported);

  TensorCXX qsrc({4}, DType_::QSigned8, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX qdst({4}, DType_::Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor qsv = qsrc.mutableCTensor();
  Tensor qdv = qdst.mutableCTensor();
  EXPECT_EQ(cast(&qsv, &qdv, DType_::Float32).err, novaCastNotSupported);
}
