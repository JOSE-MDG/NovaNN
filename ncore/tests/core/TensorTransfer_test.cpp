/**
 * @file TensorTransfer_test.cpp
 * @brief Unit tests for tensor-level host↔device transfers.
 *
 * Exercises `transf_tensor_from_host()` / `transf_tensor_from_device()`
 * (ncore/src/core/tensor.c): direction validators, rejection of tensors
 * without backing storage, and real host↔device round trips on GPU builds.
 * The bridge-level `deviceTransfer()` unknown-kind guard
 * (ncore/memory/csrc/ffi.cpp) gets direct coverage too.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

#include <tests/common/Gpu.hpp>
#include <tests/rng/Fill.hpp>
#include <tests/tensor/TensorUtils.hpp>

namespace {

using ncore::wrappers::TensorCXX;
using ncore::wrappers::unallocated;
using tests::rng::fillTensor;
using tests::tensor::expectBytesEqual;

extern "C" novaStatus_t deviceTransfer(const void *src, void *dst,
                                       TransferKind kind, size_t bytes,
                                       void *stream = nullptr);

TEST(TensorTransfer, FromHostRejectsCpuDestination) {
  novaStatus_t st{};
  TensorCXX src({4}, Float32, DEVICE_CPU, false, false, &st);
  TensorCXX dst({4}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  Tensor dv = dst.getCTensor();

  st = transf_tensor_from_host(&sv, &dv);
  EXPECT_EQ(st.err, novaTransferError);
  EXPECT_NE(st.message, nullptr);
}

TEST(TensorTransfer, FromDeviceRejectsCpuPair) {
  novaStatus_t st{};
  TensorCXX src({4}, Float32, DEVICE_CPU, false, false, &st);
  TensorCXX dst({4}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  Tensor dv = dst.getCTensor();

  st = transf_tensor_from_device(&sv, &dv);
  EXPECT_EQ(st.err, novaTransferError);
  EXPECT_NE(st.message, nullptr);
}

TEST(TensorTransfer, FromHostRejectsUnallocatedSource) {
  NOVA_TEST_REQUIRE_GPU();
  novaStatus_t st{};
  TensorCXX src(unallocated, {4}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();

  TensorCXX dst(unallocated, {4}, Float32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor dv = dst.getCTensor();

  st = transf_tensor_from_host(&sv, &dv);
  EXPECT_EQ(st.err, novaTransferError);
  EXPECT_NE(st.message, nullptr);
}

TEST(TensorTransfer, ValidatorsRejectMetaTensors) {
  novaStatus_t st{};
  TensorCXX meta({2, 2}, Float32, DEVICE_META, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor mv = meta.getCTensor();

  TensorCXX cpu({2, 2}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor cv = cpu.getCTensor();

  st = transf_tensor_from_host(&mv, &cv);
  EXPECT_EQ(st.err, novaTransferError);
  EXPECT_NE(st.message, nullptr);

  st = transf_tensor_from_device(&cv, &mv);
  EXPECT_EQ(st.err, novaTransferError);
  EXPECT_NE(st.message, nullptr);
}

TEST(TensorTransfer, DeviceTransferRejectsUnknownKind) {
  const novaStatus_t st =
      deviceTransfer(nullptr, nullptr, static_cast<TransferKind>(0), 0);

  EXPECT_EQ(st.err, novaInvalidTransfDirection);
  EXPECT_NE(st.message, nullptr);
}

TEST(TensorTransfer, FromDeviceRejectsGpuTaggedUnallocatedSource) {
  NOVA_TEST_REQUIRE_GPU();
  novaStatus_t st{};
  TensorCXX src(unallocated, {4}, Float32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();

  TensorCXX dst({4}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor dv = dst.getCTensor();

  st = transf_tensor_from_device(&sv, &dv);
  EXPECT_EQ(st.err, novaTransferError);
  EXPECT_NE(st.message, nullptr);
}

TEST(TensorTransfer, FromHostRejectsGpuTaggedUnallocatedDestination) {
  NOVA_TEST_REQUIRE_GPU();
  novaStatus_t st{};
  TensorCXX src({4}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();

  TensorCXX dst(unallocated, {4}, Float32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor dv = dst.getCTensor();

  st = transf_tensor_from_host(&sv, &dv);
  EXPECT_EQ(st.err, novaTransferError);
  EXPECT_NE(st.message, nullptr);
}

#if defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)

TEST(TensorTransfer, HostDeviceRoundTrip) {
  NOVA_TEST_REQUIRE_GPU();
  const std::vector<size_t> shape = {4, 8};
  novaStatus_t st{};

  TensorCXX host(shape, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor hv = host.getCTensor();
  EXPECT_EQ(fillTensor(hv).err, novaSuccess);

  TensorCXX dev(shape, Float32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor dv = dev.getCTensor();

  st = transf_tensor_from_host(&hv, &dv);
  ASSERT_EQ(st.err, novaSuccess);

  TensorCXX back(shape, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor bv = back.getCTensor();

  st = transf_tensor_from_device(&dv, &bv);
  ASSERT_EQ(st.err, novaSuccess);

  expectBytesEqual(hv.data.data, bv.data.data, hv.size * hv.item_size,
                   "round-trip payload");
}

TEST(TensorTransfer, ByteSizeVarietyRoundTrip) {
  NOVA_TEST_REQUIRE_GPU();
  for (const size_t count :
       {size_t{1}, size_t{63}, size_t{512}, size_t{4096}}) {
    SCOPED_TRACE(::testing::Message() << "count=" << count);
    novaStatus_t st{};

    TensorCXX host({count}, UnSigned8, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    Tensor hv = host.getCTensor();
    EXPECT_EQ(fillTensor(hv).err, novaSuccess);

    TensorCXX dev({count}, UnSigned8, DEVICE_GPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    Tensor dv = dev.getCTensor();

    st = transf_tensor_from_host(&hv, &dv);
    ASSERT_EQ(st.err, novaSuccess);

    TensorCXX back({count}, UnSigned8, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    Tensor bv = back.getCTensor();

    st = transf_tensor_from_device(&dv, &bv);
    ASSERT_EQ(st.err, novaSuccess);

    expectBytesEqual(hv.data.data, bv.data.data, hv.size * hv.item_size,
                     "round-trip payload");
  }
}

TEST(TensorTransfer, DtypeSubsetRoundTrip) {
  NOVA_TEST_REQUIRE_GPU();
  for (const DType_ dtype : {Float32, UnSigned8, Float16}) {
    SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int>(dtype));
    const std::vector<size_t> shape = {16};
    novaStatus_t st{};

    TensorCXX host(shape, dtype, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    Tensor hv = host.getCTensor();
    EXPECT_EQ(fillTensor(hv).err, novaSuccess);

    TensorCXX dev(shape, dtype, DEVICE_GPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    Tensor dv = dev.getCTensor();

    st = transf_tensor_from_host(&hv, &dv);
    ASSERT_EQ(st.err, novaSuccess);

    TensorCXX back(shape, dtype, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    Tensor bv = back.getCTensor();

    st = transf_tensor_from_device(&dv, &bv);
    ASSERT_EQ(st.err, novaSuccess);

    expectBytesEqual(hv.data.data, bv.data.data, hv.size * hv.item_size,
                     "round-trip payload");
  }
}

TEST(TensorTransfer, PinnedStagingBuffer) {
  NOVA_TEST_REQUIRE_GPU();
  const std::vector<size_t> shape = {256};
  novaStatus_t st{};

  TensorCXX pinnedSrc(shape, Float32, DEVICE_CPU, false, true, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor psv = pinnedSrc.getCTensor();
  EXPECT_EQ(fillTensor(psv).err, novaSuccess);

  TensorCXX dev(shape, Float32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor dv = dev.getCTensor();

  st = transf_tensor_from_host(&psv, &dv);
  ASSERT_EQ(st.err, novaSuccess);

  TensorCXX pinnedDst(shape, Float32, DEVICE_CPU, false, true, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor pdv = pinnedDst.getCTensor();

  st = transf_tensor_from_device(&dv, &pdv);
  ASSERT_EQ(st.err, novaSuccess);

  expectBytesEqual(psv.data.data, pdv.data.data, psv.size * psv.item_size,
                   "pinned staging payload");
}

TEST(TensorTransfer, FromDeviceRejectsUnallocatedDestination) {
  NOVA_TEST_REQUIRE_GPU();
  novaStatus_t st{};

  TensorCXX src({4}, Float32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();

  TensorCXX dst(unallocated, {4}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor dv = dst.getCTensor();

  st = transf_tensor_from_device(&sv, &dv);
  EXPECT_EQ(st.err, novaTransferError);
  EXPECT_NE(st.message, nullptr);
}

TEST(TensorTransfer, ValidatorsEnforceOnLiveBackend) {
  NOVA_TEST_REQUIRE_GPU();
  novaStatus_t st{};
  TensorCXX a({4}, Float32, DEVICE_CPU, false, false, &st);
  TensorCXX b({4}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor av = a.getCTensor();
  Tensor bv = b.getCTensor();

  st = transf_tensor_from_host(&av, &bv);
  EXPECT_EQ(st.err, novaTransferError);
  EXPECT_NE(st.message, nullptr);

  st = transf_tensor_from_device(&av, &bv);
  EXPECT_EQ(st.err, novaTransferError);
  EXPECT_NE(st.message, nullptr);
}

#endif // defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)

} // namespace
