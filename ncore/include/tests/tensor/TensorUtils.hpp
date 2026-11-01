/**
 * @file TensorUtils.hpp
 * @brief Shared comparisons and allocation helpers for the tensor suites.
 *
 * Deterministic data entry stays outside: fills come from the generator
 * wrapper at each call site.
 *
 * Ownership rule: every tensor is owned by `ncore::wrappers::TensorCXX`.
 * Unallocated tensors (e.g. `deepcopy()` destinations) are built through
 * its tagged constructors `TensorCXX(unallocated, ...)` and handed to
 * C out-param APIs via `mutableCTensor()`.
 */

#pragma once

#include <gtest/gtest.h>

#include <cstring>

#include <ncore/core/alloc.h>
#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

namespace tests::tensor {

/**
 * @brief Non-fatal bitwise comparison of two buffers with context.
 */
inline void expectBytesEqual(const void *a, const void *b, size_t bytes,
                             const char *ctx) {
  EXPECT_EQ(0, std::memcmp(a, b, bytes)) << ctx;
}

/**
 * @brief Field-by-field metadata equality including the fixed fields.
 */
inline void expectMetadataEqual(const Tensor &a, const Tensor &b) {
  ASSERT_EQ(a.ndims, b.ndims);
  for (size_t dim = 0; dim < a.ndims; ++dim) {
    EXPECT_EQ(a.shape[dim], b.shape[dim]) << "shape[" << dim << "]";
    EXPECT_EQ(a.strides[dim], b.strides[dim]) << "strides[" << dim << "]";
  }
  EXPECT_EQ(a.item_size, b.item_size);
  EXPECT_EQ(a.size, b.size);
  EXPECT_EQ(a.logical_size, b.logical_size);
  EXPECT_EQ(a.dtype, b.dtype);
  EXPECT_EQ(a.device, b.device);
  EXPECT_EQ(a.scale_, b.scale_);
  EXPECT_EQ(a.zero_point_, b.zero_point_);
  EXPECT_EQ(a.requires_grad_, b.requires_grad_);
  EXPECT_EQ(a.retain_grad_, b.retain_grad_);
  EXPECT_EQ(a.is_pinned_, b.is_pinned_);

  EXPECT_TRUE(b.is_leaf_);
  EXPECT_FALSE(b.is_view_);
  EXPECT_EQ(b.offset, size_t{0});
  EXPECT_EQ(b.version_, size_t{0});
  EXPECT_EQ(b.grad_fn_, nullptr);
}

/**
 * @brief Allocates backing storage for an unallocated gradient tensor.
 *
 * Mirrors what a post-backward gradient looks like before `deepcopy()`.
 */
inline novaStatus_t allocateGrad(Tensor &ten) {
  return safe_allocator(ten.grad->size * ten.grad->item_size, ten.grad->device,
                        ten.grad->is_pinned_, nullptr, ten.grad, true);
}

} // namespace tests::tensor
