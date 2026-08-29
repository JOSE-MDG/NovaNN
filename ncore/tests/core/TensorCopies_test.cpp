/**
 * @file TensorCopies_test.cpp
 * @brief Unit tests for tensor deep-copying.
 *
 * Exercises `deepcopy()` (ncore/src/core/copy.c): payload and metadata
 * equality across shapes, recursive gradient copying, META tensors, error
 * reporting for invalid destinations and device/dtype pairs, and
 * device-to-device copies on GPU builds.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <ncore/core/alloc.h>
#include <ncore/core/copy.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

#include "utils/TensorUtils.hpp"

namespace {

using ncore::wrappers::TensorCXX;
using ncore::wrappers::unallocated;
using tests::tensor::expectBytesEqual;
using tests::tensor::expectMetadataEqual;
using tests::tensor::fillPattern;
using tests::tensor::fillTensor;

TEST(TensorCopies, MetadataAndDataEquality) {
  novaStatus_t st{};
  TensorCXX src({3, 4, 5}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor srcView = src.getCTensor();
  fillTensor(srcView);

  TensorCXX dst(unallocated, {3, 4, 5}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&srcView, &dst.mutableCTensor(), &st);
  ASSERT_EQ(st.err, novaSuccess);

  const Tensor dcv = dst.getCTensor();
  EXPECT_TRUE(is_allocated(&dcv));
  expectMetadataEqual(srcView, dcv);
  expectBytesEqual(srcView.data.data, dcv.data.data,
                   srcView.storage->size_bytes, "deep-copied payload");
}

TEST(TensorCopies, ShapeVariantsAreCopied) {
  novaStatus_t st{};
  const std::vector<std::vector<size_t>> shapes = {
      {7}, {2, 3, 4}, std::vector<size_t>(NOVA_MAX_DIMS, 1), {1}};

  for (const auto &shape : shapes) {
    SCOPED_TRACE(::testing::Message() << "rank=" << shape.size());
    TensorCXX src(shape, Float32, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    Tensor sv = src.getCTensor();
    fillTensor(sv);

    TensorCXX dst(unallocated, shape, Float32, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);

    deepcopy(&sv, &dst.mutableCTensor(), &st);
    ASSERT_EQ(st.err, novaSuccess);

    const Tensor dcv = dst.getCTensor();
    expectMetadataEqual(sv, dcv);
    expectBytesEqual(sv.data.data, dcv.data.data, sv.storage->size_bytes,
                     "deep-copied payload");
  }

  /* Scalar variant: created through the scalar constructors. */
  TensorCXX ssrc(Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor ssv = ssrc.getCTensor();
  fillTensor(ssv);

  TensorCXX sdst(unallocated, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&ssv, &sdst.mutableCTensor(), &st);
  ASSERT_EQ(st.err, novaSuccess);

  const Tensor sdcv = sdst.getCTensor();
  expectBytesEqual(ssv.data.data, sdcv.data.data, ssv.storage->size_bytes,
                   "scalar deep-copied payload");
}

TEST(TensorCopies, GradientSubtreeDeepCopied) {
  novaStatus_t st{};
  TensorCXX src({2, 3}, Float32, DEVICE_CPU, true, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  fillTensor(sv);
  ASSERT_TRUE(sv.grad != nullptr);

  st = tests::tensor::allocateGrad(sv);
  ASSERT_EQ(st.err, novaSuccess);
  fillPattern(sv.grad->data.data, sv.grad->storage->size_bytes, 0xDEADBEEFu);

  /* Destination without a pre-existing grad: deepcopy attaches the copy. */
  TensorCXX dst(unallocated, {2, 3}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&sv, &dst.mutableCTensor(), &st);
  ASSERT_EQ(st.err, novaSuccess);

  const Tensor dcv = dst.getCTensor();
  ASSERT_TRUE(dcv.grad != nullptr);
  EXPECT_TRUE(is_allocated(dcv.grad));
  EXPECT_TRUE(dcv.grad->is_leaf_);
  expectBytesEqual(sv.grad->data.data, dcv.grad->data.data,
                   sv.grad->storage->size_bytes, "gradient payload");

  /* Independence of the grad buffers. */
  sv.grad->data.u8[0] = static_cast<uint8>(sv.grad->data.u8[0] ^ 0xFFu);
  EXPECT_NE(0, std::memcmp(sv.grad->data.data, dcv.grad->data.data,
                           sv.grad->storage->size_bytes));
}

TEST(TensorCopies, ScalarGradientBranch) {
  novaStatus_t st{};
  TensorCXX src(Float32, DEVICE_CPU, true, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  ASSERT_TRUE(is_scalar(&sv));
  ASSERT_TRUE(sv.grad != nullptr);

  st = tests::tensor::allocateGrad(sv);
  ASSERT_EQ(st.err, novaSuccess);
  fillPattern(sv.grad->data.data, sv.grad->storage->size_bytes);

  TensorCXX dst(unallocated, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&sv, &dst.mutableCTensor(), &st);
  ASSERT_EQ(st.err, novaSuccess);

  const Tensor dcv = dst.getCTensor();
  ASSERT_TRUE(dcv.grad != nullptr);
  EXPECT_TRUE(is_allocated(dcv.grad));
  expectBytesEqual(sv.grad->data.data, dcv.grad->data.data,
                   sv.grad->storage->size_bytes, "scalar gradient payload");
}

TEST(TensorCopies, CopyIsIndependentFromSource) {
  novaStatus_t st{};
  TensorCXX src({8}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  fillTensor(sv);

  TensorCXX dst(unallocated, {8}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&sv, &dst.mutableCTensor(), &st);
  ASSERT_EQ(st.err, novaSuccess);

  const size_t bytes = sv.storage->size_bytes;
  std::vector<uint8_t> snapshot(bytes);
  std::memcpy(snapshot.data(), dst.getCTensor().data.data, bytes);

  fillPattern(sv.data.data, bytes, 0xBADFACE5u);
  expectBytesEqual(snapshot.data(), dst.getCTensor().data.data, bytes,
                   "destination snapshot after source mutation");
}

/* META tensors have no backing storage: only metadata is copied. */
TEST(TensorCopies, MetaTensorMetadataOnlyCopy) {
  novaStatus_t st{};
  TensorCXX src({2, 3}, Float32, DEVICE_META, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  TensorCXX dst(unallocated, {2, 3}, Float32, DEVICE_META, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&src.mutableCTensor(), &dst.mutableCTensor(), &st);
  ASSERT_EQ(st.err, novaSuccess);

  const Tensor scv = src.getCTensor();
  const Tensor dcv = dst.getCTensor();
  EXPECT_FALSE(is_allocated(&scv));
  EXPECT_FALSE(is_allocated(&dcv));
  EXPECT_EQ(scv.storage, nullptr);
  EXPECT_EQ(dcv.storage, nullptr);
  expectMetadataEqual(scv, dcv);
}

TEST(TensorCopies, NullDestinationReportsInvalidTensor) {
  novaStatus_t st{};
  TensorCXX src({2}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();

  deepcopy(&sv, nullptr, &st);
  EXPECT_EQ(st.err, novaInvalidTensor);
  EXPECT_NE(st.message, nullptr);
}

TEST(TensorCopies, AllocatedDestinationRejected) {
  novaStatus_t st{};
  TensorCXX src({2}, Float32, DEVICE_CPU, false, false, &st);
  TensorCXX dst({2}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  Tensor dv = dst.getCTensor();

  deepcopy(&sv, &dv, &st);
  EXPECT_EQ(st.err, novaInvalidTensor);
  ASSERT_NE(st.message, nullptr);
  EXPECT_NE(std::strstr(st.message, "create_unallocated_tensor"), nullptr);
}

TEST(TensorCopies, DeviceMismatchReportsInvalidDevice) {
  novaStatus_t st{};
  TensorCXX src({2, 2}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();

  TensorCXX dst(unallocated, {2, 2}, Float32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&sv, &dst.mutableCTensor(), &st);
  EXPECT_EQ(st.err, novaInvalidDevice);
}

/* A dtype mismatch takes precedence over the device mismatch. */
TEST(TensorCopies, DeviceAndDtypeMismatchReportsInvalidDtype) {
  novaStatus_t st{};
  TensorCXX src({2, 2}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();

  TensorCXX dst(unallocated, {2, 2}, BFloat16, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&sv, &dst.mutableCTensor(), &st);
  EXPECT_EQ(st.err, novaInvalidDtype);
}

#if defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)

namespace {

/**
 * @brief Fills a device tensor by transferring a host pattern into it.
 *
 * Setup utility for GPU cases; uses the tensor-level transfer API.
 */
void fillDeviceTensor(Tensor &dev, const std::vector<size_t> &logical_shape,
                      uint32_t seed) {
  novaStatus_t st{};
  TensorCXX host(logical_shape, dev.dtype, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor hv = host.getCTensor();
  fillTensor(hv, seed);
  st = transf_tensor_from_host(&hv, &dev);
  ASSERT_EQ(st.err, novaSuccess);
}

/// Short label for parameterized dtype names.
std::string dtypeLabel(DType_ dtype) {
  switch (dtype) {
  case Float32:
    return "f32";
  case Signed32:
    return "i32";
  case Float16:
    return "f16";
  default:
    return "dt";
  }
}

} // namespace

/* Parameterized device-to-device deep copy round trip per dtype. */
class TensorCopiesDevice : public ::testing::TestWithParam<DType_> {};

TEST_P(TensorCopiesDevice, DeviceDeepCopyRoundTrip) {
  NOVA_TEST_REQUIRE_GPU();
  const DType_ dtype = GetParam();
  const std::vector<size_t> shape = {4, 8};
  SCOPED_TRACE(dtypeLabel(dtype));

  novaStatus_t st{};
  TensorCXX src(shape, dtype, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  fillDeviceTensor(sv, shape, tests::tensor::kDefaultSeed);

  TensorCXX dst(unallocated, shape, dtype, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&sv, &dst.mutableCTensor(), &st);
  ASSERT_EQ(st.err, novaSuccess);

  TensorCXX backSrc(shape, dtype, DEVICE_CPU, false, false, &st);
  TensorCXX backDst(shape, dtype, DEVICE_CPU, false, false, &st);
  Tensor bs = backSrc.getCTensor();
  Tensor bd = backDst.getCTensor();
  const Tensor dv = dst.getCTensor();

  st = transf_tensor_from_device(&sv, &bs);
  ASSERT_EQ(st.err, novaSuccess);
  st = transf_tensor_from_device(&dv, &bd);
  ASSERT_EQ(st.err, novaSuccess);

  const size_t bytes = sv.storage->size_bytes;
  expectBytesEqual(bs.data.data, bd.data.data, bytes,
                   "device deep-copied payload");

  /* Independence on device: rewrite src, destination keeps old bytes. */
  std::vector<uint8_t> snapshot(bytes);
  std::memcpy(snapshot.data(), bd.data.data, bytes);
  fillDeviceTensor(sv, shape, 0x13579BDFu);
  st = transf_tensor_from_device(&dv, &bd);
  ASSERT_EQ(st.err, novaSuccess);
  expectBytesEqual(snapshot.data(), bd.data.data, bytes,
                   "device destination after source rewrite");
}

INSTANTIATE_TEST_SUITE_P(DeviceDtypes, TensorCopiesDevice,
                         ::testing::Values(Float32, Signed32, Float16),
                         [](const ::testing::TestParamInfo<DType_> &info) {
                           switch (info.param) {
                           case Float32:
                             return std::string("f32");
                           case Signed32:
                             return std::string("i32");
                           case Float16:
                             return std::string("f16");
                           default:
                             return std::string("dt");
                           }
                         });

TEST(TensorCopies, DeviceGradientSubtree) {
  NOVA_TEST_REQUIRE_GPU();
  const std::vector<size_t> shape = {2, 3};
  novaStatus_t st{};

  TensorCXX src(shape, Float32, DEVICE_GPU, true, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  fillDeviceTensor(sv, shape, tests::tensor::kDefaultSeed);
  ASSERT_TRUE(sv.grad != nullptr);

  st = tests::tensor::allocateGrad(sv);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX gsrc(shape, Float32, DEVICE_CPU, false, false, &st);
  Tensor gv = gsrc.getCTensor();
  fillTensor(gv);
  st = transf_tensor_from_host(&gv, sv.grad);
  ASSERT_EQ(st.err, novaSuccess);

  TensorCXX dst(unallocated, shape, Float32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&sv, &dst.mutableCTensor(), &st);
  ASSERT_EQ(st.err, novaSuccess);

  const Tensor dcv = dst.getCTensor();
  ASSERT_TRUE(dcv.grad != nullptr);
  EXPECT_TRUE(is_allocated(dcv.grad));

  TensorCXX gback(shape, Float32, DEVICE_CPU, false, false, &st);
  Tensor gb = gback.getCTensor();
  st = transf_tensor_from_device(dcv.grad, &gb);
  ASSERT_EQ(st.err, novaSuccess);
  expectBytesEqual(gv.data.data, gb.data.data, gv.storage->size_bytes,
                   "device gradient payload");
}

TEST(TensorCopies, PinnedHostDeepCopy) {
  NOVA_TEST_REQUIRE_GPU();
  novaStatus_t st{};

  TensorCXX src({64}, Float32, DEVICE_CPU, false, true, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  fillTensor(sv);

  TensorCXX dst(unallocated, {64}, Float32, DEVICE_CPU, false, true, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&sv, &dst.mutableCTensor(), &st);
  ASSERT_EQ(st.err, novaSuccess);

  const Tensor dcv = dst.getCTensor();
  EXPECT_TRUE(dcv.is_pinned_);
  expectBytesEqual(sv.data.data, dcv.data.data, sv.storage->size_bytes,
                   "pinned deep-copied payload");
}

#endif // defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)

} // namespace
