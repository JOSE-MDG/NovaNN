/**
 * @file device.c
 * @brief Device-backend detection, dispatch-table setup, and inter-backend
 *        memory transfers for the core C layer.
 *
 * Implements the public device availability checks declared in device.h,
 * initialises the transfer-kind dispatch table (transf_dispatch), and
 * provides transfer_to() which routes inter-device memory copies through
 * the C-callable device_memcpy_c() wrapper.  CUDA and HIP probes are
 * delegated to backend-specific detection units.
 */

#include <ncore/cpp_ffi.h>
#include <ncore/device.h>

/**
 * @brief Return the active HIP device id.
 * @return Device id, or -1 when HIP is unavailable.
 */
extern int get_hip_device_id(void);

/**
 * @brief Return the active CUDA device id.
 * @return Device id, or -1 when CUDA is unavailable.
 */
extern int get_cuda_device_id(void);

/**
 * @brief Lookup table to dispatch the type of transfer according to the type of
 * attribute of the device.
 *
 */
TransferKind transf_dispatch[3][3] = {0};

__attribute__((constructor)) static inline void init_transf_dispatch() {

  transf_dispatch[DEVICE_GPU][DEVICE_CPU] = deviceMemcpyDeviceToHost;
  transf_dispatch[DEVICE_CPU][DEVICE_GPU] = deviceMemcpyHostToDevice;
  transf_dispatch[DEVICE_GPU][DEVICE_GPU] = deviceMemcpyDeviceToDevice;
}

/**
 * @brief Check whether any GPU device backend is available.
 *
 * @param kind Requested backend kind.
 * @param verbose If true, backend probes may print runtime diagnostics.
 * @return true when the requested backend reports an available device.
 */
bool is_device_available(DeviceKind kind, bool verbose) {
  switch (kind) {
  case CUDA_DEVICE:
    return is_cuda_device_available(verbose);
  case HIP_DEVICE:
    return is_hip_device_available(verbose);
  case NULL_DEVICE:
  default:
    return false;
  }
}

/**
 * @brief Check whether CUDA should be selected as the active backend.
 *
 * @return true when CUDA reports an available device.
 */
bool is_cuda_available(void) { return is_cuda_device_available(false); }

/**
 * @brief Check whether HIP should be selected as the active backend.
 *
 * @return true when HIP reports an available device.
 */
bool is_hip_available(void) { return is_hip_device_available(false); }

/**
 * @brief Transfer memory between device backends.
 *
 * Looks up the correct copy direction from the dispatch table
 * (transf_dispatch) and forwards the request to device_memcpy_c.
 *
 * @param dst       Target device placement.
 * @param src       Source device placement.
 * @param src_buf   Pointer to the source buffer.
 * @param dst_buf   Pointer to the destination buffer.
 * @param is_pinned Whether the host-side buffer is pinned/page-locked.
 * @param bytes     Number of bytes to transfer.
 * @return DeviceStatus with code 0 on success, or an error status.
 */
DeviceStatus transfer_to(Device dst, Device src, const void *src_buf,
                         void *dst_buf, bool is_pinned, size_t bytes) {
  TransferKind kind = transf_dispatch[dst][src];
  return device_memcpy_c(src_buf, dst_buf, is_pinned, kind, bytes);
}

/**
 * @brief Return the active device id (CUDA/HIP).
 *
 * @return Device id, or -1 when is unavailable
 */
int get_device_id(void) {
  if (is_cuda_available()) {
    return get_cuda_device_id();
  }
  if (is_hip_available()) {
    return get_hip_device_id();
  }
  return -1;
}
