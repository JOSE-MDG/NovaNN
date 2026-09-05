/**
 * @file ContiguousLayoutKernel.hip
 * @brief HIP kernels materializing contiguous tensor layouts.
 *
 * @details
 * Mirror of @c ContiguousLayoutKernel.cu: grid-stride kernels for
 * the 1-D collapsed and N-D cases, plus a device-to-device fast path
 * for 1-D tensors. Launch geometry comes from
 * @ref resolve_launch_config() fed with the detected device
 * properties, so both backends size grids identically.
 *
 * @see launch_config.h  Shared grid/block resolution.
 * @see ContiguousLayoutKernel.h  Launcher declaration.
 */

#include <cstddef>
#include <hip/hip_runtime.h>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/native/kernels/launch_config.h>
#include <ncore/tensor.h>

#include "ContiguousLayoutKernel.h"
#include "DetectHipDeviceInfo.hpp"
#include "memory/HipIO.hpp"

namespace {

/**
 * @brief Copy a one-dimensional tensor via device-to-device transfer.
 *
 * @details
 * Host-side helper (not a kernel): validates the dimension count,
 * moves the buffer with @ref hipTransfer() and recomputes the
 * destination strides for the contiguous layout.
 *
 * @param[in]  src  Source tensor. Must have exactly one dimension.
 * @param[out] dst  Destination tensor receiving the data.
 * @param[out] st   Receives the operation result.
 */
void contiguousKernelOneDimTensor(const Tensor *src, Tensor *dst,
                                  novaStatus_t *st) {

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
 * @brief Gather a collapsed one-dimensional view into @p dst.
 *
 * @details
 * Grid-stride kernel: each thread copies every element whose index
 * equals @c threadIdx.x + @c blockIdx.x * @c blockDim.x modulo the
 * total grid stride. Strided sources are gathered element by
 * element; already-contiguous ones take the device-to-device fast
 * path through @ref hipTransfer().
 *
 * @param[in] stride    Source byte stride between elements.
 * @param[in] count     Element count to gather.
 * @param[in] item_size Bytes per element; a full element is copied.
 * @param[in] src_data  Source byte buffer in device memory.
 * @param[out] dst_data Destination byte buffer in device memory.
 */
__global__ void contiguousKernelOneDimCV(size_t stride, size_t count,
                                         size_t item_size,
                                         const unsigned char *src_data,
                                         unsigned char *dst_data) {
  const size_t idx =
      (size_t)threadIdx.x + (size_t)blockDim.x * (size_t)blockIdx.x;
  const size_t step = (size_t)blockDim.x * (size_t)gridDim.x;

  for (size_t item = idx; item < count; item += step) {
    const unsigned char *s = src_data + (item * stride);
    unsigned char *d = dst_data + (item * item_size);
    for (size_t b = 0; b < item_size; ++b) {
      d[b] = s[b];
    }
  }
}

/**
 * @brief Gather a multi-dimensional collapsed view into @p dst.
 *
 * @details
 * Grid-stride kernel over the flat element index. Each thread
 * rebuilds its coordinates with
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
                                     const unsigned char *src_data,
                                     unsigned char *dst_data) {
  const size_t idx =
      (size_t)threadIdx.x + (size_t)blockDim.x * (size_t)blockIdx.x;
  const size_t step = (size_t)blockDim.x * (size_t)gridDim.x;

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
} // namespace

/**
 * @brief Materialize a contiguous copy of @p src into @p dst on HIP.
 *
 * @details
 * Mirror of @ref launchCudaContiguousKernel(): already-contiguous
 * sources become views; 1-D tensors take the device-to-device fast
 * path; anything else is collapsed and copied by a grid-stride
 * kernel sized with @ref resolve_launch_config() from the detected
 * device properties.
 *
 * @param[in]  src  Source tensor in HIP device memory.
 * @param[out] dst  Destination tensor in HIP device memory.
 *
 * @return @ref novaStatus_t with the launch outcome, or the property
 *         query error when the device caps cannot be read.
 *
 * @see resolve_launch_config()
 * @see launchCudaContiguousKernel()  CUDA mirror.
 */
novaStatus_t launchHipContiguousKernel(const Tensor *src, Tensor *dst) {

  novaStatus_t st{};

  if (is_contiguous(src)) {
    *dst = create_view(src, src->shape, src->ndims, &st);
    st.err = novaSuccess;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  }

  if (src->ndims == 1) {
    contiguousKernelOneDimTensor(src, dst, &st);
    return st;
  }

  const CollapsedView cv = collapse(src);

  hipDetectedDeviceProps_t properties = getHipDeviceProperties(&st);
  if (st.err != novaSuccess) {
    return st;
  }
  const LaunchConfig cfg = resolve_launch_config(
      src->size, src->item_size,
      static_cast<unsigned int>(properties.maxThreadsPerBlock),
      static_cast<unsigned int>(properties.multiProcessorCount));
  dim3 blocks(cfg.threads, 1, 1);
  dim3 grid(cfg.blocks, 1, 1);

  if (cv.ndims == 1) {
    if (cv.strides[0] != src->item_size) {
      contiguousKernelOneDimCV<<<grid, blocks>>>(
          cv.strides[0], src->size, src->item_size, src->data.data,
          dst->data.data);
    } else {
      st = hipTransfer(src->storage->size_bytes,
                       DeviceMemcpyKind::deviceMemcpyDeviceToDevice,
                       src->data.v, dst->data.v);
      if (st.err != novaSuccess) {
        return st;
      }
    }
  } else {
    contiguousKernelNDim<<<grid, blocks>>>(cv, src->offset, src->size,
                                           src->item_size, src->data.data,
                                           dst->data.data);
  }
  st.err = novaSuccess;
  st.message = nova_get_error_msg(st.err, nullptr);
  return st;
}
