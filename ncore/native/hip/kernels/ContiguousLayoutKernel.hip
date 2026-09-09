/**
 * @file ContiguousLayoutKernel.hip
 * @brief HIP kernels materializing contiguous tensor layouts.
 *
 * @details
 * Mirror of @c ContiguousLayoutKernel.cu: vectorized gathers, a
 * shared-tile path for column-contiguous 2-D views, and a
 * device-to-device fast path for 1-D tensors. Launch geometry comes
 * from @ref ncore::heuristics::kernels::resolveLaunchConfig() fed with
 * the detected device properties, so both backends size grids
 * identically.
 *
 * @see launch_config.hh  Shared grid/block resolution.
 * @see ContiguousLayoutKernel.h  Launcher declaration.
 */

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/heuristics/kernels/launch_config.hh>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/tensor.h>

#include "ContiguousLayoutKernel.h"
#include "DetectHipDeviceInfo.hpp"
#include "ffi.hpp"
#include "memory/HipIO.hpp"

namespace {

/// Every accessed address stays a multiple of @p vB.
inline bool stridesVecOk(const CollapsedView &cv, size_t srcOff, size_t vB) {
  if (srcOff % vB != 0U) {
    return false;
  }
  for (size_t i = 0; i < cv.ndims; ++i) {
    if (cv.strides[i] % vB != 0U) {
      return false;
    }
  }
  return true;
}

/// Widest provable transaction in bytes, 0 for the byte fallback.
inline size_t vecWidthBytes(size_t itemSize, size_t totalBytes,
                            const void *sp, const void *dp, size_t srcOff,
                            const CollapsedView &cv) {
  namespace hk = ncore::heuristics::kernels;
  if (itemSize == 0U) {
    return 0U;
  }
  const size_t sA = hk::pointerAlign(sp);
  const size_t dA = hk::pointerAlign(dp);
  for (size_t vB : {16U, 8U, 4U, 2U}) {
    if (itemSize > vB || vB % itemSize != 0U) {
      continue;
    }
    if (totalBytes % vB != 0U) {
      continue;
    }
    if (sA < vB || dA < vB) {
      continue;
    }
    if (!stridesVecOk(cv, srcOff, vB)) {
      continue;
    }
    return vB;
  }
  return 0U;
}

/// Column-contiguous 2-D view with single-element transactions.
inline bool tile2DApplicable(const CollapsedView &cv, size_t itemSize,
                             const void *sp, const void *dp, size_t srcOff) {
  namespace hk = ncore::heuristics::kernels;
  if (cv.ndims != 2U) {
    return false;
  }
  if (itemSize != 1U && itemSize != 2U && itemSize != 4U && itemSize != 8U) {
    return false;
  }
  if (hk::pointerAlign(sp) < itemSize || hk::pointerAlign(dp) < itemSize) {
    return false;
  }
  if (srcOff % itemSize != 0U) {
    return false;
  }
  if (cv.strides[0] != itemSize) {
    return false;
  }
  return cv.strides[1] % itemSize == 0U;
}

void contiguousKernelOneDimTensor(const Tensor *restrict src,
                                  Tensor *restrict dst, novaStatus_t *st) {

  if (src->ndims != 1) {
    st->err = novaInvalidTensor;
    st->message = "Cannot perform contiguous operation with a non "
                  "one-dimensional tensor";
    return;
  }

  *st = hipTransfer(src->storage->size_bytes,
                    DeviceMemcpyKind::deviceMemcpyDeviceToDevice, src->data.v,
                    dst->data.v);
  if (st->err != novaSuccess) {
    return;
  }
  compute_tensor_strides_(dst, src->ndims, src->shape, src->item_size);
  st->err = novaSuccess;
  st->message = nova_get_error_msg(st->err, nullptr);
}

/**
 * @brief Strided 1-D gather with one vector transaction per element.
 *
 * @param[in] strideVec Stride in transactions, pre-divided on host.
 * @param[in] srcOffVec Base offset in transactions, pre-divided.
 * @param[in] count     Transaction count.
 */
template <typename VecT>
__global__ void contiguousKernelOneDimCVVec(size_t strideVec, size_t srcOffVec,
                                            size_t count,
                                            const VecT *restrict src_data,
                                            VecT *restrict dst_data) {
  const size_t idx =
      static_cast<size_t>(threadIdx.x) +
      (static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x));
  const size_t step =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  for (size_t item = idx; item < count; item += step) {
    dst_data[item] = __ldg(src_data + srcOffVec + (item * strideVec));
  }
}

/**
 * @brief Gather a collapsed one-dimensional view into @p dst.
 *
 * @details
 * Byte fallback when no vector width proves. Grid-stride kernel:
 * each thread copies every element whose index equals @c threadIdx.x
 * + @c blockIdx.x * @c blockDim.x modulo the total grid stride.
 *
 * @param[in] stride    Source byte stride between elements.
 * @param[in] src_offset Source byte offset into @p src_data.
 * @param[in] count     Element count to gather.
 * @param[in] item_size Bytes per element; a full element is copied.
 * @param[in] src_data  Source byte buffer in device memory.
 * @param[out] dst_data Destination byte buffer in device memory.
 */
__global__ void contiguousKernelOneDimCV(size_t stride, size_t src_offset,
                                         size_t count, size_t item_size,
                                         const unsigned char *restrict src_data,
                                         unsigned char *restrict dst_data) {
  const size_t idx =
      static_cast<size_t>(threadIdx.x) +
      (static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x));
  const size_t step =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  for (size_t item = idx; item < count; item += step) {
    const unsigned char *s = src_data + src_offset + (item * stride);
    unsigned char *d = dst_data + (item * item_size);
    for (size_t b = 0; b < item_size; ++b) {
      d[b] = __ldg(s + b);
    }
  }
}

/**
 * @brief N-D gather with one vector transaction per element.
 *
 * @details
 * Strides and base offset arrive pre-divided into transaction units,
 * so the hot loop holds no division. Destination stays dense.
 */
template <typename VecT>
__global__ void contiguousKernelNDimVec(CollapsedView cv, size_t srcOffVec,
                                        size_t count,
                                        const VecT *restrict src_data,
                                        VecT *restrict dst_data) {
  const size_t idx =
      static_cast<size_t>(threadIdx.x) +
      (static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x));
  const size_t step =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  for (size_t item = idx; item < count; item += step) {
    size_t offset = srcOffVec;
    coords_t coords = {0};
    compute_coords_from_linear_index_(item, cv.ndims, cv.shape, coords);

#pragma unroll
    for (size_t dim = 0; dim < cv.ndims; ++dim) {
      offset += coords[dim] * cv.strides[dim];
    }

    dst_data[item] = __ldg(src_data + offset);
  }
}

/**
 * @brief Gather a multi-dimensional collapsed view into @p dst.
 *
 * @details
 * Byte fallback when no vector width proves. Grid-stride kernel over
 * the flat element index. Each thread rebuilds its coordinates with
 * @ref compute_coords_from_linear_index_() (thread-local, no shared
 * state) and accumulates the source byte offset from the collapsed
 * strides.
 *
 * @param[in] cv         Collapsed view (by value) describing the
 *                       source layout.
 * @param[in] src_offset Source byte offset into @p src_data.
 * @param[in] count      Element count to gather.
 * @param[in] item_size  Bytes per element; a full element is copied.
 * @param[in] src_data   Source byte buffer in device memory.
 * @param[out] dst_data  Destination byte buffer in device memory.
 */
__global__ void contiguousKernelNDim(CollapsedView cv, size_t src_offset,
                                     size_t count, size_t item_size,
                                     const unsigned char *restrict src_data,
                                     unsigned char *restrict dst_data) {
  const size_t idx =
      static_cast<size_t>(threadIdx.x) +
      (static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x));
  const size_t step =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  for (size_t item = idx; item < count; item += step) {
    size_t offset = src_offset;
    coords_t coords = {0};
    compute_coords_from_linear_index_(item, cv.ndims, cv.shape, coords);

#pragma unroll
    for (size_t dim = 0; dim < cv.ndims; ++dim) {
      offset += coords[dim] * cv.strides[dim];
    }

    const unsigned char *s = src_data + offset;
    unsigned char *d = dst_data + (item * item_size);
    for (size_t b = 0; b < item_size; ++b) {
      d[b] = __ldg(s + b);
    }
  }
}

/**
 * @brief 2-D column-contiguous gather through a shared tile.
 *
 * @details
 * Each 32x8 thread group stages one 32x32 tile: threads read down
 * the contiguous stride (coalesced) into padded shared rows, then
 * write dense rows back (coalesced). One element per transaction;
 * the caller proves the width equals @c item_size. Edge tiles
 * predicate both sides; the tile grid strides when it exceeds the
 * grid.
 */
template <typename VecT>
__global__ void contiguousKernel2DTileVec(size_t rows, size_t cols, size_t s0v,
                                          size_t s1v, size_t srcOffV,
                                          const VecT *restrict src_data,
                                          VecT *restrict dst_data) {
  __shared__ VecT tile[32][33];
  const size_t tileCols = (cols + 31U) / 32U;
  const size_t numTiles = ((rows + 31U) / 32U) * tileCols;
  const size_t flat =
      (static_cast<size_t>(blockIdx.y) * static_cast<size_t>(gridDim.x)) +
      static_cast<size_t>(blockIdx.x);
  const size_t step =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(gridDim.y);
  const size_t tx = static_cast<size_t>(threadIdx.x);
  const size_t ty = static_cast<size_t>(threadIdx.y);

  for (size_t t = flat; t < numTiles; t += step) {
    const size_t baseI = (t / tileCols) * 32U;
    const size_t baseJ = (t % tileCols) * 32U;
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k) {
      const size_t rLd = baseI + tx;
      const size_t cLd = baseJ + (ty * 4U) + k;
      VecT v{};
      if (rLd < rows && cLd < cols) {
        v = __ldg(src_data + srcOffV + rLd * s0v + cLd * s1v);
      }
      tile[tx][(ty * 4U) + k] = v;
    }
    __syncthreads();
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k) {
      const size_t r = baseI + (ty * 4U) + k;
      const size_t c = baseJ + tx;
      if (r < rows && c < cols) {
        dst_data[(r * cols) + c] = tile[(ty * 4U) + k][tx];
      }
    }
    __syncthreads();
  }
}

template <typename VecT>
void launchOneDimCVVec(size_t strideBytes, size_t srcOffBytes, size_t count,
                       size_t itemSize, const unsigned char *src,
                       unsigned char *dst, dim3 grid, dim3 blocks) {
  constexpr size_t VB = sizeof(VecT);
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
  contiguousKernelOneDimCVVec<VecT><<<grid, blocks>>>(
      strideBytes / VB, srcOffBytes / VB, (count * itemSize) / VB,
      reinterpret_cast<const VecT *>(src), reinterpret_cast<VecT *>(dst));
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

template <typename VecT>
void launchNDimVec(const CollapsedView &cv, size_t srcOff, size_t count,
                   size_t itemSize, const unsigned char *src,
                   unsigned char *dst, dim3 grid, dim3 blocks) {
  constexpr size_t VB = sizeof(VecT);
  CollapsedView cvv = cv;
  for (size_t i = 0; i < cvv.ndims; ++i) {
    cvv.strides[i] /= VB;
  }
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
  contiguousKernelNDimVec<VecT><<<grid, blocks>>>(
      cvv, srcOff / VB, (count * itemSize) / VB,
      reinterpret_cast<const VecT *>(src), reinterpret_cast<VecT *>(dst));
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

template <typename VecT>
void launch2DTileVec(size_t rows, size_t cols, size_t s0, size_t s1,
                     size_t srcOff, const unsigned char *src,
                     unsigned char *dst) {
  constexpr size_t VB = sizeof(VecT);
  const size_t tileCols = (cols + 31U) / 32U;
  const size_t tileRows = (rows + 31U) / 32U;
  const dim3 blocks(32U, 8U);
  const dim3 grid(
      static_cast<unsigned int>(tileCols > 65535U ? 65535U : tileCols),
      static_cast<unsigned int>(tileRows > 65535U ? 65535U : tileRows));
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
  contiguousKernel2DTileVec<VecT><<<grid, blocks>>>(
      rows, cols, s0 / VB, s1 / VB, srcOff / VB,
      reinterpret_cast<const VecT *>(src), reinterpret_cast<VecT *>(dst));
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

} // namespace

/**
 * @brief Materialize a contiguous copy of @p src into @p dst on HIP.
 *
 * @details
 * Mirror of @ref launchCudaContiguousKernel(): already-contiguous
 * sources become views; 1-D tensors take the device-to-device fast
 * path; column-contiguous 2-D views take the shared tile; anything
 * else is collapsed and gathered with the widest proven vector
 * width, sized with
 * @ref ncore::heuristics::kernels::resolveLaunchConfig() from the
 * detected device properties.
 *
 * @param[in]  src  Source tensor in HIP device memory.
 * @param[out] dst  Destination tensor in HIP device memory.
 *
 * @return @ref novaStatus_t with the launch outcome, or the property
 *         query error when the device caps cannot be read.
 *
 * @see ncore::heuristics::kernels::resolveLaunchConfig()
 * @see launchCudaContiguousKernel()  CUDA mirror.
 */
novaStatus_t launchHipContiguousKernel(const Tensor *restrict src,
                                       Tensor *restrict dst) {

  novaStatus_t st{};

  if (is_contiguous(src)) {
    *dst = create_view(src, src->shape, src->ndims, &st);
    return st;
  }

  if (src->ndims == 1 && src->strides[0] == src->item_size &&
      src->offset == 0) {
    contiguousKernelOneDimTensor(src, dst, &st);
    return st;
  }

  const CollapsedView cv = collapse(src);

  if (tile2DApplicable(cv, src->item_size, src->data.v, dst->data.v,
                       src->offset)) {
    const size_t rows = cv.shape[0];
    const size_t cols = cv.shape[1];
    switch (src->item_size) {
    case 8U:
      launch2DTileVec<uint2>(rows, cols, cv.strides[0], cv.strides[1],
                             src->offset, src->data.data, dst->data.data);
      break;
    case 4U:
      launch2DTileVec<uint32_t>(rows, cols, cv.strides[0], cv.strides[1],
                                src->offset, src->data.data, dst->data.data);
      break;
    case 2U:
      launch2DTileVec<uint16_t>(rows, cols, cv.strides[0], cv.strides[1],
                                src->offset, src->data.data, dst->data.data);
      break;
    default:
      launch2DTileVec<unsigned char>(rows, cols, cv.strides[0], cv.strides[1],
                                     src->offset, src->data.data,
                                     dst->data.data);
      break;
    }
    st.err = novaSuccess;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  }

  hipDetectedDeviceProps_t properties = getHipDeviceProperties(&st);
  if (st.err != novaSuccess) {
    return st;
  }
  namespace hk = ncore::heuristics::kernels;
  hk::ElementWiseParams hp{};
  hp.device = hk::capsFromDetected(properties);
  hp.numElements = src->size;
  hp.inputItemSize = static_cast<uint32_t>(src->item_size);
  hp.outputItemSize = static_cast<uint32_t>(dst->item_size);
  hp.inputAlignBytes = hk::pointerAlign(src->data.v);
  hp.outputAlignBytes = hk::pointerAlign(dst->data.v);
  hp.packedElemsPerUnit = 1;
  hp.inputContiguous = is_contiguous(src);
  hp.outputContiguous = is_contiguous(dst);
  hp.numInputs = 1;
  hp.arithmeticIntensity = 0.0F;
  hp.reuseAfter = false;
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(hp);
  dim3 blocks(cfg.threads.x, cfg.threads.y, cfg.threads.z);
  dim3 grid(cfg.blocks.x, cfg.blocks.y, cfg.blocks.z);

  if (cv.ndims == 1) {
    if (cv.strides[0] != src->item_size) {
      const size_t totalBytes = src->storage->size_bytes;
      switch (vecWidthBytes(src->item_size, totalBytes, src->data.v,
                            dst->data.v, src->offset, cv)) {
      case 16U:
        launchOneDimCVVec<uint4>(cv.strides[0], src->offset, src->size,
                                 src->item_size, src->data.data, dst->data.data,
                                 grid, blocks);
        break;
      case 8U:
        launchOneDimCVVec<uint2>(cv.strides[0], src->offset, src->size,
                                 src->item_size, src->data.data, dst->data.data,
                                 grid, blocks);
        break;
      case 4U:
        launchOneDimCVVec<uint32_t>(cv.strides[0], src->offset, src->size,
                                    src->item_size, src->data.data,
                                    dst->data.data, grid, blocks);
        break;
      case 2U:
        launchOneDimCVVec<uint16_t>(cv.strides[0], src->offset, src->size,
                                    src->item_size, src->data.data,
                                    dst->data.data, grid, blocks);
        break;
      default:
        contiguousKernelOneDimCV<<<grid, blocks>>>(
            cv.strides[0], src->offset, src->size, src->item_size,
            src->data.data, dst->data.data);
        break;
      }
    } else {
      st = hipTransfer(src->storage->size_bytes,
                       DeviceMemcpyKind::deviceMemcpyDeviceToDevice,
                       src->data.v, dst->data.v);
      if (st.err != novaSuccess) {
        return st;
      }
    }
  } else {
    const size_t totalBytes = src->storage->size_bytes;
    switch (vecWidthBytes(src->item_size, totalBytes, src->data.v, dst->data.v,
                          src->offset, cv)) {
    case 16U:
      launchNDimVec<uint4>(cv, src->offset, src->size, src->item_size,
                           src->data.data, dst->data.data, grid, blocks);
      break;
    case 8U:
      launchNDimVec<uint2>(cv, src->offset, src->size, src->item_size,
                           src->data.data, dst->data.data, grid, blocks);
      break;
    case 4U:
      launchNDimVec<uint32_t>(cv, src->offset, src->size, src->item_size,
                              src->data.data, dst->data.data, grid, blocks);
      break;
    case 2U:
      launchNDimVec<uint16_t>(cv, src->offset, src->size, src->item_size,
                              src->data.data, dst->data.data, grid, blocks);
      break;
    default:
      contiguousKernelNDim<<<grid, blocks>>>(cv, src->offset, src->size,
                                             src->item_size, src->data.data,
                                             dst->data.data);
      break;
    }
  }
  st.err = novaSuccess;
  st.message = nova_get_error_msg(st.err, nullptr);
  return st;
}
