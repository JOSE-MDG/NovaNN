/**
 * @file TensorUtils.hpp
 * @brief Shared deterministic fixtures for the tensor copy/transfer suites.
 *
 * Deterministic pattern generation, bitwise comparison, metadata
 * equality, and GPU availability guards shared by the copy/transfer
 * suites.
 *
 * Ownership rule: every tensor is owned by `ncore::wrappers::TensorCXX`.
 * Unallocated tensors (e.g. `deepcopy()` destinations) are built through
 * its tagged constructors — `TensorCXX(unallocated, ...)` — and handed to
 * C out-param APIs via `mutableCTensor()`.
 */

#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include <ncore/core/alloc.h>
#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

namespace tests::tensor {

/// Default seed for the deterministic byte pattern ("NOVA" in ASCII).
constexpr uint32_t kDefaultSeed = 0x4E4F5641u;

/**
 * @brief Fills @p buf with a deterministic LCG byte pattern.
 *
 * Same inputs produce the same bytes on every platform; no global state.
 *
 * @param[out] buf   Destination buffer.
 * @param[in]  bytes Size in bytes.
 * @param[in]  seed  LCG seed.
 */
inline void fillPattern(void *buf, size_t bytes, uint32_t seed = kDefaultSeed) {
  auto *out = static_cast<uint8_t *>(buf);
  uint32_t state = seed;
  for (size_t i = 0; i < bytes; ++i) {
    state = (state * 1664525u) + 1013904223u;
    out[i] = static_cast<uint8_t>(state >> 24);
  }
}

/**
 * @brief Fills the whole data buffer of an allocated tensor.
 *
 * @param[in,out] ten  Allocated tensor whose payload is overwritten.
 * @param[in]     seed LCG seed.
 */
inline void fillTensor(Tensor &ten, uint32_t seed = kDefaultSeed) {
  EXPECT_TRUE(is_allocated(&ten));
  fillPattern(ten.data.data, ten.storage->size_bytes, seed);
}

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

/**
 * @brief True when a GPU backend is compiled in and a device is present.
 */
inline bool gpuAvailable() {
#if defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)
  return is_cuda_available() || is_hip_available();
#else
  return false;
#endif
}

} // namespace tests::tensor

/**
 * @brief Skips the current test when no usable GPU backend/device.
 *
 * Must be invoked as the first statement of a GPU-guarded test body so
 * `GTEST_SKIP` returns from the test itself. Probing also warms the one-shot
 * device-detection cache consulted by `copy_device_buffer()`.
 */
#define NOVA_TEST_REQUIRE_GPU()                                                \
  do {                                                                         \
    if (!tests::tensor::gpuAvailable()) {                                      \
      GTEST_SKIP() << "GPU backend/device not available";                      \
    }                                                                          \
  } while (0)
