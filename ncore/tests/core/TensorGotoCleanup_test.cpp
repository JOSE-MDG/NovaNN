/**
 * @file TensorGotoCleanup_test.cpp
 * @brief Pins for the goto-cleanup refactor.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include <ncore/core/alloc.h>
#include <ncore/core/copy.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

#include <tests/common/Gpu.hpp>
#include <tests/rng/Fill.hpp>
#include <tests/tensor/TensorUtils.hpp>

using ncore::wrappers::TensorCXX;
using ncore::wrappers::unallocated;
using tests::tensor::allocateGrad;

namespace {
bool isZeroed(const Tensor &t) {
  Tensor zero{};
  return std::memcmp(&t, &zero, sizeof(Tensor)) == 0;
}
} // namespace

TEST(TensorGotoCleanup, TransposeOutOfRangeReturnsZeroed) {
  novaStatus_t st{};
  TensorCXX src({2, 3}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  ASSERT_TRUE(is_allocated(&sv));

  Tensor view = transpose(&sv, &st, 0, 5);
  EXPECT_NE(st.err, novaSuccess);
  EXPECT_TRUE(isZeroed(view));

  /* Source untouched: still allocated and collectable exactly once. */
  EXPECT_TRUE(is_allocated(&sv));
  EXPECT_EQ(collect(&sv).err, novaSuccess);
}

TEST(TensorGotoCleanup, PermuteDuplicateReturnsZeroed) {
  novaStatus_t st{};
  TensorCXX src({2, 3}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();

  int dims[NOVA_MAX_DIMS] = {0, 0};
  Tensor view = permute(&sv, &st, dims);
  EXPECT_NE(st.err, novaSuccess);
  EXPECT_TRUE(isZeroed(view));

  EXPECT_TRUE(is_allocated(&sv));
  EXPECT_EQ(collect(&sv).err, novaSuccess);
}

TEST(TensorGotoCleanup, ContiguousKernelFailurePathZeroes) {
  novaStatus_t st{};
  TensorCXX src({2, 3}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  /* Force non-contiguity by transposing, then contiguous() copies. */
  Tensor t = transpose(&sv, &st, 0, 1);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor c = contiguous(&t, &st);
  ASSERT_EQ(st.err, novaSuccess);
  EXPECT_TRUE(is_contiguous(&c));
  EXPECT_EQ(collect(&c).err, novaSuccess);
  EXPECT_EQ(collect(&sv).err, novaSuccess);
}

TEST(TensorGotoCleanup, DeepcopyFailureLeavesDstOwned) {
  novaStatus_t st{};
  TensorCXX src({2, 3}, Float32, DEVICE_CPU, true, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  ASSERT_TRUE(sv.grad != nullptr);
  ASSERT_EQ(allocateGrad(sv).err, novaSuccess);

  /* Force the recursive deepcopy(sv.grad, ...) to fail by corrupting the
     grad's own storage reservation: safe_allocator(0) returns
     novaInvalidValue before allocating the destination grad buffer. */
  ASSERT_TRUE(sv.grad->storage != nullptr);
  sv.grad->storage->handle.requested_bytes = 0;

  TensorCXX dst(unallocated, {2, 3}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  deepcopy(&sv, &dst.mutableCTensor(), &st);
  EXPECT_NE(st.err, novaSuccess);

  const Tensor dcv = dst.getCTensor();
  EXPECT_TRUE(is_allocated(&dcv));

  EXPECT_EQ(collect(&dst.mutableCTensor()).err, novaSuccess);
}
