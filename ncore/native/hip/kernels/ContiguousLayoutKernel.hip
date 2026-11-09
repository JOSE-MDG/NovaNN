/**
 * @file ContiguousLayoutKernel.hip
 * @brief HIP kernels materializing contiguous tensor layouts.
 *
 * @details
 * Routing stays in the launcher: 1-D contiguous sources move through
 * the device-to-device fast path, column-contiguous 2-D views take the
 * shared-tile kernel, and everything else is collapsed and gathered.
 * Two run-time gates split the gather paths:
 *
 * - Counts below 2^32 index through uint32 arithmetic and take
 *   four-lane group kernels. The strided 1-D case stages 16 B of
 *   output per thread behind one wide transaction when the dense
 *   destination proves that alignment; ranks 2 to 4 additionally
 *   replace the per-dimension coordinate division with the
 *   multiply-shift reciprocal table from @ref ncore::magic and advance
 *   the remaining lanes with an odometer carry, so a 64-bit division
 *   never executes on this path.
 * - Deeper counts, ranks past the specialization, odd item sizes, or
 *   views the vector probes cannot prove fall back to the single-lane
 *   kernels, which rebuild coordinates from native division.
 *
 * Launch geometry comes from
 * @ref ncore::heuristics::kernels::resolveLaunchConfig() fed with a
 * @ref ncore::heuristics::kernels::LayoutTransformParams description
 * and the detected device properties, so no per-kernel magic numbers
 * are needed. Block shapes stay wavefront agnostic: the sizer derives
 * occupancy facts from @c hipFuncGetAttributes(), which reports the
 * register and block limits of the current wave (32 on RDNA, 64 on
 * CDNA parts).
 *
 * @see launch_config.hh  Shared grid/block resolution.
 * @see magic_division.hh  Per-launch reciprocal tables.
 * @see ContiguousLayoutKernel.h  Launcher declarations.
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/heuristics/kernels/launch_config.hh>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/magic_division.hh>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/memory/ffi.hpp>
#include <ncore/tensor.h>

#include "ContiguousLayoutKernel.h"
#include "DetectHipDeviceInfo.hpp"
#include "memory/HipIO.hpp"

namespace hk = ncore::heuristics::kernels;
namespace nm = ncore::magic;

static_assert(hk::LAYOUT_MAX_RANK >= NOVA_MAX_DIMS,
              "LayoutShape must describe every tensor rank");

namespace {

/**
 * One element per transaction, 0 for the byte fallback.
 *
 * Both vector kernels index their input in element space: the N-D kernel
 * rebuilds coordinates from @c cv.shape, whose product is the element
 * count, and the 1-D kernel walks @c item * stride. A transaction wider
 * than one element would cover several elements while the coordinates
 * advance one at a time; a narrower one would copy a fragment at the
 * element stride. Alignment and strides only prove the access is legal,
 * not that the index space still matches.
 */
inline size_t elementTxnBytes(size_t itemSize, const void *sp, const void *dp,
                              size_t srcOff, const CollapsedView &cv) {
  if (itemSize == 0U || itemSize > 8U) {
    return 0U;
  }
  if ((itemSize & (itemSize - 1U)) != 0U) {
    return 0U;
  }
  if (hk::pointerAlign(static_cast<const unsigned char *>(sp) + srcOff) <
          itemSize ||
      hk::pointerAlign(dp) < itemSize) {
    return 0U;
  }
  for (size_t i = 0; i < cv.ndims; ++i) {
    if (cv.strides[i] % itemSize != 0U) {
      return 0U;
    }
  }
  return itemSize;
}

/// Column-contiguous 2-D view with single-element transactions.
inline bool tile2DApplicable(const CollapsedView &cv, size_t itemSize,
                             const void *sp, const void *dp, size_t srcOff) {
  if (cv.ndims != 2U) {
    return false;
  }
  if (itemSize != 1U && itemSize != 2U && itemSize != 4U && itemSize != 8U) {
    return false;
  }
  if (hk::pointerAlign(static_cast<const unsigned char *>(sp) + srcOff) <
          itemSize ||
      hk::pointerAlign(dp) < itemSize) {
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
                                  Tensor *restrict dst, novaStatus_t *st,
                                  hipStream_t stream) {
  *st = hipTransfer(src->size * src->item_size,
                    DeviceMemcpyKind::deviceMemcpyDeviceToDevice,
                    src->data.data + src->offset, dst->data.data, stream);
  if (st->err != novaSuccess) {
    return;
  }
  compute_tensor_strides_(dst, src->ndims, src->shape, src->item_size);
  st->err = novaSuccess;
  st->message = nova_get_error_msg(st->err, nullptr);
}

/// Lanes per thread in the group kernels; the sizer budgets the same
/// count through hk::FAST_ELEMS_PER_THREAD, so one pass covers the grid.
constexpr uint32_t kGroupLanes = 4U;

/// Maximum rank the specialized N-D group kernels bake.
constexpr uint32_t kGroupMaxRank = 4U;

/// Element counts this large index through size_t kernels instead.
constexpr uint64_t kU32ElemCeil = (1ULL << 32);

/**
 * @brief Store one fully valid output group in wide transactions.
 *
 * @details
 * The launcher proves the destination base alignment (the dense side
 * fills from byte zero), so the staged lane block issues as the widest
 * transaction its width allows: 4 B for byte lanes up to two 16 B
 * transactions for double lanes. On this backend those compile to
 * @c flat_dwordx4 style stores whose addresses the wavefronts coalesce
 * per lane. The cast widens the alignment requirement beyond what the
 * variable type proves, so it stays behind diagnostic suppression with
 * that contract documented here once.
 */
__device__ __forceinline__ void
storeGroupWide(unsigned char *dst, const unsigned char (&vals)[kGroupLanes]) {
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
  *reinterpret_cast<unsigned int *>(dst) =
      *reinterpret_cast<const unsigned int *>(vals);
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

__device__ __forceinline__ void
storeGroupWide(unsigned short *dst, const unsigned short (&vals)[kGroupLanes]) {
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
  *reinterpret_cast<uint2 *>(dst) = *reinterpret_cast<const uint2 *>(vals);
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

__device__ __forceinline__ void
storeGroupWide(unsigned int *dst, const unsigned int (&vals)[kGroupLanes]) {
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
  *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(vals);
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

__device__ __forceinline__ void
storeGroupWide(uint2 *dst, const uint2 (&vals)[kGroupLanes]) {
  uint4 *const out = reinterpret_cast<uint4 *>(dst);
  out[0] = uint4{vals[0].x, vals[0].y, vals[1].x, vals[1].y};
  out[1] = uint4{vals[2].x, vals[2].y, vals[3].x, vals[3].y};
}

/**
 * @brief Store one group as per-lane transactions.
 *
 * @details
 * The partial-tail shape and the destination without proven group
 * alignment end here: each valid lane stores its own transaction.
 */
template <typename VecT>
__device__ __forceinline__ void storeGroupLanes(VecT *dst,
                                                const VecT (&vals)[kGroupLanes],
                                                uint32_t base, uint32_t count) {
#pragma unroll
  for (uint32_t k = 0U; k < kGroupLanes; ++k) {
    if (base + k < count) {
      dst[k] = vals[k];
    }
  }
}

/**
 * @brief Gather a strided one-dimensional view in four-lane groups.
 *
 * @details
 * Each thread moves four consecutive elements: separate gathered loads
 * (the source can never hold adjacent elements, the stride is strictly
 * above the element size) and one wide store for the dense block when
 * the whole group is valid and the destination proved the alignment.
 * After the grid-stride loop has covered the tail groups the counts
 * stay under 2^32, so the lane, group, and index arithmetic runs in
 * uint32 without a wrap: the group lockstep never products past the
 * element count by more than three.
 *
 * @param[in]  strideVec  Source stride in transaction units; at least 2.
 * @param[in]  srcOffVec  Source byte offset in transaction units.
 * @param[in]  count      Element count; strictly below @c 2^32.
 * @param[in]  src_data   Source buffer in device memory.
 * @param[out] dst_data   Destination buffer in device memory.
 */
template <typename VecT, bool wide>
__global__ void contiguousKernelOneDimGroup(uint32_t strideVec,
                                            size_t srcOffVec, uint32_t count,
                                            const VecT *restrict src_data,
                                            VecT *restrict dst_data) {
  const uint32_t tid =
      (static_cast<uint32_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const uint32_t nt = blockDim.x * gridDim.x;
  const uint32_t groups = (count >> 2U) + ((count & 3U) != 0U ? 1U : 0U);

  for (uint32_t g = tid; g < groups; g += nt) {
    const uint32_t base = g * kGroupLanes;
    const size_t off0 = srcOffVec + (static_cast<size_t>(base) * strideVec);
    VecT vals[kGroupLanes];
    bool whole = true;
    vals[0] = src_data[off0];
#pragma unroll
    for (uint32_t k = 1U; k < kGroupLanes; ++k) {
      if (base + k >= count) {
        whole = false;
        break;
      }
      vals[k] = src_data[off0 + (static_cast<size_t>(k) * strideVec)];
    }
    if (wide && whole) {
      storeGroupWide(dst_data + base, vals);
    } else {
      storeGroupLanes(dst_data + base, vals, base, count);
    }
  }
}

/**
 * @brief Bulk contiguous copy for counts past the uint32 gate.
 *
 * @details
 * Fallback for tensors of 4 GiB elements and beyond: size_t indexing
 * throughout, one streamed element per thread.
 *
 * @copydetails contiguousKernelOneDimGroup()
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
    dst_data[item] = src_data[srcOffVec + (item * strideVec)];
  }
}

/**
 * @brief Gather a collapsed one-dimensional view into @p dst_data.
 *
 * @details
 * Byte fallback when no vector width proves: same grid-stride shape
 * as the group kernel, one streamed byte at a time.
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
      d[b] = s[b];
    }
  }
}

/**
 * @brief Packed collapsed-view facts for the group N-D gather.
 *
 * @details
 * Everything the coordinate rebuild needs, held narrow so the launch
 * parameter block stays broadcast friendly: extents below 2^32 (the
 * uint32 gate bounds the whole view), per-dimension strides in
 * transaction units, and the element count.
 */
struct NDimGroupParams {
  uint32_t shape[kGroupMaxRank];     ///< Extents per collapsed dimension.
  uint32_t strideVec[kGroupMaxRank]; ///< Strides in transaction units.
  uint32_t count;                    ///< Element count, below 2^32.
};

/**
 * @brief Advance collapsed coordinates one item, odometer style.
 *
 * @details
 * Walks from the innermost dimension outward: the carry resets a full
 * extent and moves one unit along the next one out. The offset tracks
 * the same walk in transaction units; the wrap subtraction restores
 * the full-extent contribution the increment skipped. A carried walk
 * covering every dimension can only land back on the base layout,
 * which the group bound forbids for valid lanes.
 */
template <uint32_t rank>
__device__ __forceinline__ void
advanceCoords(uint32_t (&coords)[rank], const NDimGroupParams &p, size_t *off) {
#pragma unroll
  for (uint32_t d = rank; d-- > 0U;) {
    if (++coords[d] < p.shape[d]) {
      *off += p.strideVec[d];
      return;
    }
    coords[d] = 0U;
    *off -= static_cast<size_t>(p.shape[d] - 1U) * p.strideVec[d];
  }
}

/**
 * @brief Gather ranks 2 to 4 in four-lane groups via the magic table.
 *
 * @details
 * Each thread stages one group of four consecutive elements. The lane
 * 0 offset decodes from the linear group index through the
 * multiply-shift table (one 32-bit multiply and shift per dimension,
 * innermost first, replacing one 64-bit division and modulo), and the
 * remaining lanes advance those coordinates odometer style, which only
 * pays the carry walk instead of a fresh decode. Gathered loads and
 * the dense store follow the same streaming shape as the 1-D group
 * kernel. All packed multiplication stays in 64 bits, so no offset
 * accumulation wraps.
 *
 * @param[in]  p          Packed shape and strides, see @c NDimGroupParams.
 * @param[in]  magic      Reciprocal table packed for every dimension.
 * @param[in]  srcOffVec  Source byte offset in transaction units.
 * @param[in]  src_data   Source buffer in device memory.
 * @param[out] dst_data   Destination buffer in device memory.
 */
template <typename VecT, bool wide, uint32_t rank>
__global__ void
contiguousKernelNDimGroup(NDimGroupParams p, nm::DivMagicSet magic,
                          size_t srcOffVec, const VecT *restrict src_data,
                          VecT *restrict dst_data) {
  static_assert(rank >= 2U && rank <= kGroupMaxRank,
                "specialized ranks are baked per instantiation");
  const uint32_t tid =
      (static_cast<uint32_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const uint32_t nt = blockDim.x * gridDim.x;
  const uint32_t groups = (p.count >> 2U) + ((p.count & 3U) != 0U ? 1U : 0U);

  for (uint32_t g = tid; g < groups; g += nt) {
    const uint32_t base = g * kGroupLanes;
    size_t off = srcOffVec;
    uint32_t coords[rank];
    uint32_t x = base;
#pragma unroll
    for (uint32_t d = rank; d-- > 0U;) {
      const uint32_t extent = p.shape[d];
      const uint32_t q =
          nm::packed_div32(magic.mul[d], magic.shift[d], magic.wide[d], x);
      coords[d] = x - (q * extent);
      x = q;
      off += static_cast<size_t>(coords[d]) * p.strideVec[d];
    }

    VecT vals[kGroupLanes];
    vals[0] = src_data[off];
    bool whole = true;
#pragma unroll
    for (uint32_t k = 1U; k < kGroupLanes; ++k) {
      if (base + k >= p.count) {
        whole = false;
        break;
      }
      advanceCoords(coords, p, &off);
      vals[k] = src_data[off];
    }
    if (wide && whole) {
      storeGroupWide(dst_data + base, vals);
    } else {
      storeGroupLanes(dst_data + base, vals, base, p.count);
    }
  }
}

/**
 * @brief Single-lane N-D gather for the deep fallback.
 *
 * @details
 * Covers ranks past the specialization and counts past the uint32
 * gate: size_t throughout, coordinates from native division per
 * element, one streamed transaction per element.
 *
 * @param[in] cv         Collapsed view (by value) describing the
 *                       source layout.
 * @param[in] srcOffVec  Source byte offset in transaction units.
 * @param[in] count      Element count to gather.
 * @param[in] src_data   Source buffer in device memory.
 * @param[out] dst_data  Destination buffer in device memory.
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

    dst_data[item] = src_data[offset];
  }
}

/**
 * @brief Gather a multi-dimensional collapsed view into @p dst_data.
 *
 * @details
 * Byte fallback when no vector width proves. Grid-stride kernel over
 * the flat element index. Each thread rebuilds its coordinates with
 * @ref compute_coords_from_linear_index_() (thread-local, no shared
 * state) and accumulates the source byte offset from the collapsed
 * strides. The view travels by value: @c CollapsedView is plain
 * @c size_t storage, so the device never sees a @c Tensor.
 *
 * @param[in] cv         Collapsed view (by value) describing the
 *                       source layout.
 * @param[in] src_offset Source byte offset into @p src_data.
 * @param[in] count      Element count to gather.
 * @param[in] item_size  Bytes per element; a full element is copied.
 * @param[in] src_data  Source byte buffer in device memory.
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
      d[b] = s[b];
    }
  }
}

/**
 * @brief 2-D column-contiguous gather through a shared tile.
 *
 * @details
 * Each 32x8 thread group stages one 32x32 tile: threads read down
 * the contiguous stride into padded LDS rows, then write dense rows
 * back. One element per transaction; the caller proves the width
 * equals @c item_size. Edge tiles predicate both sides; the tile grid
 * strides when it exceeds the grid. The 33-element row pitch keeps the
 * row-to-column read pattern spread across LDS channels, whose bank
 * width makes a 32 pitch hit the same channel per row.
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
        v = src_data[srcOffV + (rLd * s0v) + (cLd * s1v)];
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
                       unsigned char *dst, dim3 grid, dim3 threads,
                       hipStream_t stream) {
  constexpr size_t vB = sizeof(VecT);
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
  contiguousKernelOneDimCVVec<VecT><<<grid, threads, 0, stream>>>(
      strideBytes / vB, srcOffBytes / vB, (count * itemSize) / vB,
      reinterpret_cast<const VecT *>(src), reinterpret_cast<VecT *>(dst));
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

/**
 * @brief Launch the four-lane 1-D group kernel for one vector width.
 *
 * @details
 * Gates checked host-side before this call: the count fits uint32 and
 * the stride fits uint32 transaction units, so both parameter casts
 * are exact.
 */
template <typename VecT, bool wide>
void launchOneDimGroup(size_t strideBytes, size_t srcOffBytes, size_t count,
                       size_t itemSize, const unsigned char *src,
                       unsigned char *dst, dim3 grid, dim3 threads,
                       hipStream_t stream) {
  constexpr size_t vB = sizeof(VecT);
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
  contiguousKernelOneDimGroup<VecT, wide><<<grid, threads, 0, stream>>>(
      static_cast<uint32_t>(strideBytes / vB), srcOffBytes / vB,
      static_cast<uint32_t>((count * itemSize) / vB),
      reinterpret_cast<const VecT *>(src), reinterpret_cast<VecT *>(dst));
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

template <typename VecT>
void launchNDimVec(const CollapsedView &cv, size_t srcOff, size_t count,
                   size_t itemSize, const unsigned char *src,
                   unsigned char *dst, dim3 grid, dim3 threads,
                   hipStream_t stream) {
  constexpr size_t vB = sizeof(VecT);
  CollapsedView cvv = cv;
  for (size_t i = 0; i < cvv.ndims; ++i) {
    cvv.strides[i] /= vB;
  }
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
  contiguousKernelNDimVec<VecT><<<grid, threads, 0, stream>>>(
      cvv, srcOff / vB, (count * itemSize) / vB,
      reinterpret_cast<const VecT *>(src), reinterpret_cast<VecT *>(dst));
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

/**
 * @brief Launch the rank-specialized N-D group kernel for one width.
 *
 * @details
 * Gates checked host-side before this call: the rank fits the baked
 * specializations (else the caller launches the deep fallback), all
 * strides fit uint32 transaction units, and every extent resolved a
 * magic pair.
 */
template <typename VecT, bool wide>
void launchNDimGroup(uint32_t rank, const NDimGroupParams &p,
                     const nm::DivMagicSet &magic, size_t srcOffVec,
                     const unsigned char *src, unsigned char *dst, dim3 grid,
                     dim3 threads, hipStream_t stream) {
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif
  switch (rank) {
  case 2U:
    contiguousKernelNDimGroup<VecT, wide, 2U><<<grid, threads, 0, stream>>>(
        p, magic, srcOffVec, reinterpret_cast<const VecT *>(src),
        reinterpret_cast<VecT *>(dst));
    break;
  case 3U:
    contiguousKernelNDimGroup<VecT, wide, 3U><<<grid, threads, 0, stream>>>(
        p, magic, srcOffVec, reinterpret_cast<const VecT *>(src),
        reinterpret_cast<VecT *>(dst));
    break;
  default:
    contiguousKernelNDimGroup<VecT, wide, 4U><<<grid, threads, 0, stream>>>(
        p, magic, srcOffVec, reinterpret_cast<const VecT *>(src),
        reinterpret_cast<VecT *>(dst));
    break;
  }
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

template <typename VecT>
void launch2DTileVec(size_t rows, size_t cols, size_t s0, size_t s1,
                     size_t srcOff, const unsigned char *src,
                     unsigned char *dst, hipStream_t stream) {
  constexpr size_t vB = sizeof(VecT);
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
  contiguousKernel2DTileVec<VecT><<<grid, blocks, 0, stream>>>(
      rows, cols, s0 / vB, s1 / vB, srcOff / vB,
      reinterpret_cast<const VecT *>(src), reinterpret_cast<VecT *>(dst));
#ifdef _GNUC_CLANG_
#pragma GCC diagnostic pop
#endif
}

/// Compiled facts of one gather kernel, queried once per instantiation.
template <typename Kernel> hk::KernelAttrs kernelAttrs(Kernel *func) {
  static const hk::KernelAttrs attrs = [func] {
    hk::KernelAttrs a{};
    hipFuncAttributes q{};
    if (hipFuncGetAttributes(&q, reinterpret_cast<const void *>(func)) ==
        hipSuccess) {
      a.regsPerThread = static_cast<uint32_t>(q.numRegs);
      a.staticSmemBytes = q.sharedSizeBytes;
      a.maxThreadsPerBlockKernel = static_cast<uint32_t>(q.maxThreadsPerBlock);
    }
    return a;
  }();
  return attrs;
}

/// Attrs of the rank-specialized group kernel baked for @p rank.
template <typename VecT, bool wide>
hk::KernelAttrs ndimGroupAttrs(uint32_t rank) noexcept {
  switch (rank) {
  case 2U:
    return kernelAttrs(contiguousKernelNDimGroup<VecT, wide, 2U>);
  case 3U:
    return kernelAttrs(contiguousKernelNDimGroup<VecT, wide, 3U>);
  default:
    return kernelAttrs(contiguousKernelNDimGroup<VecT, wide, 4U>);
  }
}

/**
 * @brief Attrs of the gather kernel for one vector width and routing.
 *
 * @details
 * The sizer reads register and block limits from these facts, so this
 * selector and the launch branches must keep the same routing: byte
 * width, one- or multi-dimensional collapsed view, four-lane group
 * kernel applicability, and the collapsed rank. @p wide rides as a
 * template parameter because it picks the kernel instantiation.
 */
template <typename VecT>
hk::KernelAttrs typedGroupAttrs(bool oneDim, bool wide, bool groupOk,
                                uint32_t rank) noexcept {
  if (oneDim) {
    if (groupOk) {
      return wide ? kernelAttrs(contiguousKernelOneDimGroup<VecT, true>)
                  : kernelAttrs(contiguousKernelOneDimGroup<VecT, false>);
    }
    return kernelAttrs(contiguousKernelOneDimCVVec<VecT>);
  }
  if (groupOk) {
    return wide ? ndimGroupAttrs<VecT, true>(rank)
                : ndimGroupAttrs<VecT, false>(rank);
  }
  return kernelAttrs(contiguousKernelNDimVec<VecT>);
}

/**
 * @brief Attrs of the exact kernel the launch branch below selects.
 *
 * @details
 * The sizer reads register and block limits from these facts, so this
 * selector and the launch branches must keep the same routing: byte
 * width, one- or multi-dimensional collapsed view, group kernel
 * applicability (four-lane paths), the destination proven wide store,
 * and the collapsed rank.
 */
hk::KernelAttrs selectGatherAttrs(bool oneDim, size_t vecBytes, bool wide,
                                  bool groupOk, uint32_t rank) noexcept {
  if (vecBytes == 0U) {
    return oneDim ? kernelAttrs(contiguousKernelOneDimCV)
                  : kernelAttrs(contiguousKernelNDim);
  }
  switch (vecBytes) {
  case 8U:
    return typedGroupAttrs<uint2>(oneDim, wide, groupOk, rank);
  case 4U:
    return typedGroupAttrs<uint32_t>(oneDim, wide, groupOk, rank);
  case 2U:
    return typedGroupAttrs<uint16_t>(oneDim, wide, groupOk, rank);
  default:
    return typedGroupAttrs<unsigned char>(oneDim, wide, groupOk, rank);
  }
}

/**
 * @brief Wide-store proof from the dense destination base alignment.
 *
 * @details
 * The group store issues the widest transaction the lane width
 * stages: 16 B for 4 B and 8 B lanes, 8 B for 2 B lanes, 4 B for byte
 * lanes. Anything narrower falls back to per-lane stores inside the
 * same kernel.
 */
inline bool wideGroupStore(size_t vecBytes, uint32_t dstAlign) noexcept {
  if (vecBytes >= 4U) {
    return dstAlign >= 16U;
  }
  if (vecBytes == 2U) {
    return dstAlign >= 8U;
  }
  return dstAlign >= 4U;
}

/**
 * Map the pending launch error into status (also resets the sticky error).
 */
inline novaStatus_t checkLaunch() {
  if (hipGetLastError() != hipSuccess) {
    novaStatus_t s{};
    s.err = novaKernelLaunchError;
    s.message = nova_get_error_msg(s.err, nullptr);
    return s;
  }
  novaStatus_t s{};
  s.err = novaSuccess;
  s.message = nova_get_error_msg(s.err, nullptr);
  return s;
}

} // namespace

/**
 * @brief Materialize a contiguous copy of @p src into @p dst on HIP.
 *
 * @details
 * Already-contiguous inputs are handled by the caller before dispatch
 * (see tensor.c); this launcher assumes a non-contiguous source and
 * always materializes a copy: 1-D tensors take the device-to-device
 * fast path; column-contiguous 2-D views take the shared tile;
 * anything else is collapsed and gathered. Counts under 2^32 route to
 * the four-lane group kernels (uint32 indexing, the strided 1-D case
 * stages one wide transaction per thread, ranks 2 to 4 decode
 * coordinates through the multiply-shift table), the rest to the
 * single-lane fallbacks. Geometry comes from
 * @ref ncore::heuristics::kernels::resolveLaunchConfig() from a
 * @ref ncore::heuristics::kernels::LayoutTransformParams description
 * and the detected device properties.
 *
 * @param[in]  src  Source tensor in HIP device memory.
 * @param[out] dst  Destination tensor in HIP device memory.
 *
 * @return @ref novaStatus_t with the launch outcome
 *         (@ref novaKernelLaunchError when @c hipGetLastError reports
 *         a launch failure), or the property query error when the
 *         device caps cannot be read.
 *
 * @see ncore::heuristics::kernels::resolveLaunchConfig()
 * @see launchCudaContiguousKernel()  CUDA mirror.
 */
novaStatus_t launchHipContiguousKernel(const Tensor *restrict src,
                                       Tensor *restrict dst,
                                       hipStream_t stream) {

  novaStatus_t st{};

  if (src->size == 0) {
    st.err = novaSuccess;
    return st;
  }

  if (src->ndims == 1 && src->strides[0] == src->item_size &&
      src->offset == 0) {
    contiguousKernelOneDimTensor(src, dst, &st, stream);
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
                             src->offset, src->data.data, dst->data.data,
                             stream);
      break;
    case 4U:
      launch2DTileVec<uint32_t>(rows, cols, cv.strides[0], cv.strides[1],
                                src->offset, src->data.data, dst->data.data,
                                stream);
      break;
    case 2U:
      launch2DTileVec<uint16_t>(rows, cols, cv.strides[0], cv.strides[1],
                                src->offset, src->data.data, dst->data.data,
                                stream);
      break;
    default:
      launch2DTileVec<unsigned char>(rows, cols, cv.strides[0], cv.strides[1],
                                     src->offset, src->data.data,
                                     dst->data.data, stream);
      break;
    }
    st = checkLaunch();
    return st;
  }

  auto props = getHipDeviceProperties(&st);
  if (st.err != novaSuccess) {
    return st;
  }
  hk::LayoutTransformParams lp{};
  lp.device = hk::capsFromDetected(props);
  lp.layout.ndim = static_cast<uint8_t>(cv.ndims);
  for (size_t i = 0; i < cv.ndims; ++i) {
    lp.layout.shape[i] = static_cast<int64_t>(cv.shape[i]);
    lp.layout.inStrides[i] =
        static_cast<int64_t>(cv.strides[i] / src->item_size);
  }
  int64_t dense = 1;
  for (size_t i = cv.ndims; i-- > 0;) {
    lp.layout.outStrides[i] = dense;
    dense *= static_cast<int64_t>(cv.shape[i]);
  }
  for (uint8_t i = 0; i < lp.layout.ndim; ++i) {
    lp.perm[i] = i;
  }
  lp.itemSize = static_cast<uint32_t>(src->item_size);

  const size_t viewBytes = src->size * src->item_size;
  if (cv.ndims == 1 && cv.strides[0] == src->item_size) {
    return hipTransfer(viewBytes, DeviceMemcpyKind::deviceMemcpyDeviceToDevice,
                       src->data.data + src->offset, dst->data.data, stream);
  }

  const size_t vecBytes = elementTxnBytes(src->item_size, src->data.v,
                                          dst->data.v, src->offset, cv);
  const bool oneDim = (cv.ndims == 1);
  const uint32_t rank = static_cast<uint32_t>(cv.ndims);

  // Group kernel gates, kept in sync with selectGatherAttrs above.
  const bool countSmall = src->size < kU32ElemCeil;
  const bool wide = wideGroupStore(vecBytes, hk::pointerAlign(dst->data.v));
  bool oneDimGroupOk = false;
  if (oneDim && countSmall && vecBytes != 0U) {
    oneDimGroupOk = (cv.strides[0] / vecBytes) <= 0xFFFFFFFFULL;
  }
  bool ndGroupOk = false;
  NDimGroupParams gParams{};
  nm::DivMagicSet gMagic{};
  if (!oneDim && countSmall && vecBytes != 0U && rank >= 2U &&
      rank <= kGroupMaxRank) {
    bool ok = true;
    nm::DivMagic table[nm::MAGIC_MAX_NDIMS];
    for (uint32_t i = 0U; i < rank; ++i) {
      const uint64_t strideVec =
          static_cast<uint64_t>(cv.strides[i]) / vecBytes;
      if (strideVec > 0xFFFFFFFFULL) {
        ok = false;
        break;
      }
      table[i] = nm::make_magic_divisor(static_cast<uint32_t>(cv.shape[i]));
      ok = ok && (table[i].valid != 0U);
    }
    if (ok) {
      gParams.count = static_cast<uint32_t>(src->size);
      for (uint32_t i = 0U; i < rank; ++i) {
        gParams.shape[i] = static_cast<uint32_t>(cv.shape[i]);
        gParams.strideVec[i] = static_cast<uint32_t>(cv.strides[i] / vecBytes);
      }
      ndGroupOk = nm::pack_magic_set(&gMagic, table, rank);
    }
  }
  lp.kernel = selectGatherAttrs(oneDim, vecBytes, wide,
                                oneDimGroupOk || ndGroupOk, rank);

  thread_local hk::FullConfigCache cache;
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(lp, &cache);
  dim3 threads(cfg.threads.x, cfg.threads.y, cfg.threads.z);
  dim3 grid(cfg.blocks.x, cfg.blocks.y, cfg.blocks.z);

  if (oneDim) {
    if (oneDimGroupOk) {
      if (vecBytes == 8U) {
        if (wide) {
          launchOneDimGroup<uint2, true>(cv.strides[0], src->offset, src->size,
                                         src->item_size, src->data.data,
                                         dst->data.data, grid, threads, stream);
        } else {
          launchOneDimGroup<uint2, false>(
              cv.strides[0], src->offset, src->size, src->item_size,
              src->data.data, dst->data.data, grid, threads, stream);
        }
      } else if (vecBytes == 4U) {
        if (wide) {
          launchOneDimGroup<uint32_t, true>(
              cv.strides[0], src->offset, src->size, src->item_size,
              src->data.data, dst->data.data, grid, threads, stream);
        } else {
          launchOneDimGroup<uint32_t, false>(
              cv.strides[0], src->offset, src->size, src->item_size,
              src->data.data, dst->data.data, grid, threads, stream);
        }
      } else if (vecBytes == 2U) {
        if (wide) {
          launchOneDimGroup<uint16_t, true>(
              cv.strides[0], src->offset, src->size, src->item_size,
              src->data.data, dst->data.data, grid, threads, stream);
        } else {
          launchOneDimGroup<uint16_t, false>(
              cv.strides[0], src->offset, src->size, src->item_size,
              src->data.data, dst->data.data, grid, threads, stream);
        }
      } else { // vecBytes == 1U, proven by oneDimGroupOk
        if (wide) {
          launchOneDimGroup<unsigned char, true>(
              cv.strides[0], src->offset, src->size, src->item_size,
              src->data.data, dst->data.data, grid, threads, stream);
        } else {
          launchOneDimGroup<unsigned char, false>(
              cv.strides[0], src->offset, src->size, src->item_size,
              src->data.data, dst->data.data, grid, threads, stream);
        }
      }
    } else {
      switch (vecBytes) {
      case 8U:
        launchOneDimCVVec<uint2>(cv.strides[0], src->offset, src->size,
                                 src->item_size, src->data.data, dst->data.data,
                                 grid, threads, stream);
        break;
      case 4U:
        launchOneDimCVVec<uint32_t>(cv.strides[0], src->offset, src->size,
                                    src->item_size, src->data.data,
                                    dst->data.data, grid, threads, stream);
        break;
      case 2U:
        launchOneDimCVVec<uint16_t>(cv.strides[0], src->offset, src->size,
                                    src->item_size, src->data.data,
                                    dst->data.data, grid, threads, stream);
        break;
      case 1U:
        launchOneDimCVVec<uint8_t>(cv.strides[0], src->offset, src->size,
                                   src->item_size, src->data.data,
                                   dst->data.data, grid, threads, stream);
        break;
      default:
        contiguousKernelOneDimCV<<<grid, threads, 0, stream>>>(
            cv.strides[0], src->offset, src->size, src->item_size,
            src->data.data, dst->data.data);
        break;
      }
    }
  } else {
    if (ndGroupOk) {
      const size_t srcOffVec = src->offset / vecBytes;
      if (vecBytes == 8U) {
        if (wide) {
          launchNDimGroup<uint2, true>(rank, gParams, gMagic, srcOffVec,
                                       src->data.data, dst->data.data, grid,
                                       threads, stream);
        } else {
          launchNDimGroup<uint2, false>(rank, gParams, gMagic, srcOffVec,
                                        src->data.data, dst->data.data, grid,
                                        threads, stream);
        }
      } else if (vecBytes == 4U) {
        if (wide) {
          launchNDimGroup<uint32_t, true>(rank, gParams, gMagic, srcOffVec,
                                          src->data.data, dst->data.data, grid,
                                          threads, stream);
        } else {
          launchNDimGroup<uint32_t, false>(rank, gParams, gMagic, srcOffVec,
                                           src->data.data, dst->data.data, grid,
                                           threads, stream);
        }
      } else if (vecBytes == 2U) {
        if (wide) {
          launchNDimGroup<uint16_t, true>(rank, gParams, gMagic, srcOffVec,
                                          src->data.data, dst->data.data, grid,
                                          threads, stream);
        } else {
          launchNDimGroup<uint16_t, false>(rank, gParams, gMagic, srcOffVec,
                                           src->data.data, dst->data.data, grid,
                                           threads, stream);
        }
      } else { // vecBytes == 1U, proven by ndGroupOk
        if (wide) {
          launchNDimGroup<unsigned char, true>(rank, gParams, gMagic, srcOffVec,
                                               src->data.data, dst->data.data,
                                               grid, threads, stream);
        } else {
          launchNDimGroup<unsigned char, false>(
              rank, gParams, gMagic, srcOffVec, src->data.data, dst->data.data,
              grid, threads, stream);
        }
      }
    } else {
      switch (vecBytes) {
      case 8U:
        launchNDimVec<uint2>(cv, src->offset, src->size, src->item_size,
                             src->data.data, dst->data.data, grid, threads,
                             stream);
        break;
      case 4U:
        launchNDimVec<uint32_t>(cv, src->offset, src->size, src->item_size,
                                src->data.data, dst->data.data, grid, threads,
                                stream);
        break;
      case 2U:
        launchNDimVec<uint16_t>(cv, src->offset, src->size, src->item_size,
                                src->data.data, dst->data.data, grid, threads,
                                stream);
        break;
      case 1U:
        launchNDimVec<uint8_t>(cv, src->offset, src->size, src->item_size,
                               src->data.data, dst->data.data, grid, threads,
                               stream);
        break;
      default:
        contiguousKernelNDim<<<grid, threads, 0, stream>>>(
            cv, src->offset, src->size, src->item_size, src->data.data,
            dst->data.data);
        break;
      }
    }
  }
  st = checkLaunch();
  return st;
}
