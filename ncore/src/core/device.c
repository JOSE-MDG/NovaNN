/**
 * @file device.c
 * @brief Device-backend detection, dispatch-table setup, and
 *        inter-backend memory transfers for the core C layer.
 *
 * @details
 * Glue layer for the public device API declared in @ref device.h.
 * Detection is delegated to backend-specific modules
 * (@c DetectCudaDevice.cpp, @c DetectHipDevice.cpp); transfers are
 * routed through @ref transfer_to(), which resolves the copy
 * direction from the startup-initialized @ref transf_dispatch table
 * and forwards to the C-callable @c deviceTransfer() wrapper.  This
 * indirection means application code never needs to @c #include CUDA
 * or HIP headers directly.
 *
 * Device detection runs once and the result is cached in
 * @ref atomic_device_detection_done and
 * @ref atomic_detected_device_kind: a process only ever uses a single
 * GPU vendor (CUDA or HIP), so repeated runtime probes would buy
 * nothing.  Writes go through @ref runtime_flags_mtx guarded by a
 * platform one-shot initializer; after startup the table and flags
 * are read-only, so concurrent reads need no locking.
 *
 * @see device.h              Public API declarations.
 * @see DetectCudaDevice.cpp  CUDA detection implementation.
 * @see DetectHipDevice.cpp   HIP detection implementation.
 * @see status.h              novaStatus_t struct.
 * @see ffi.hpp               C-callable deviceTransfer() wrapper.
 */

#include <stdatomic.h>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/mtx.h>

/**
 * @brief Transfer memory between address spaces via the active GPU
 *        backend.
 *
 * @details
 * Defined in @c ncore/memory/csrc/ffi.cpp; dispatches to CUDA or HIP
 * based on the runtime-detected GPU kind.
 *
 * @param[in]  src   Pointer to the source buffer.
 * @param[out] dst   Pointer to the destination buffer.
 * @param[in]  kind  Copy direction (@ref TransferKind).
 * @param[in]  bytes Number of bytes to transfer.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, or an appropriate error code on failure.
 */
extern novaStatus_t deviceTransfer(const void *src, void *dst,
                                   TransferKind kind, size_t bytes);

#ifdef NOVA_HAS_CUDA

/**
 * @brief Probe the CUDA runtime for an available GPU device.
 *
 * @details
 * Defined in @c ncore/native/cuda/DetectCudaDevice.cpp.
 *
 * @param log      If @c true, print detection results to stdout.
 * @param verbose  If @c true, print detailed device properties.
 *
 * @return @c true when a CUDA-capable device is found.
 */
extern bool isCudaDeviceAvailable(bool log, bool verbose);

/**
 * @brief Return the 0-based device index of the detected CUDA GPU.
 *
 * @details
 * Defined in @c ncore/native/cuda/DetectCudaDevice.cpp; returns the
 * id set during the most recent detection probe.
 *
 * @return 0-based device index, or @c -1 if no CUDA device was detected.
 */
extern int getCudaDeviceId();

/**
 * @brief Print detailed CUDA device properties to stdout.
 *
 * @details
 * Defined in @c ncore/native/cuda/DetectCudaDeviceInfo.cpp.
 *
 * @param verbose  If @c true, print the full property block.  If
 *                 @c false, print a concise summary.
 *
 * @return @ref novaStatus_t with the result of the detection.
 */
extern novaStatus_t printCudaDeviceInfo(bool verbose);

#endif /* NOVA_HAS_CUDA */

#ifdef NOVA_HAS_HIP

/**
 * @brief Probe the HIP runtime for an available GPU device.
 *
 * @details
 * Defined in @c ncore/native/hip/DetectHipDevice.cpp.
 *
 * @param log      If @c true, print detection results to stdout.
 * @param verbose  If @c true, print detailed device properties.
 *
 * @return @c true when a HIP-capable device is found.
 */
extern bool isHipDeviceAvailable(bool log, bool verbose);

/**
 * @brief Return the 0-based device index of the detected HIP GPU.
 *
 * @details
 * Defined in @c ncore/native/hip/DetectHipDevice.cpp; returns the
 * id set during the most recent detection probe.
 *
 * @return 0-based device index, or @c -1 if no HIP device was detected.
 */
extern int getHipDeviceId();

/**
 * @brief Print detailed HIP device properties to stdout.
 *
 * @details
 * Defined in @c ncore/native/hip/DetectHipDeviceInfo.cpp.
 *
 * @param verbose  If @c true, print the full property block.  If
 *                 @c false, print a concise summary.
 *
 * @return @ref novaStatus_t with the result of the detection.
 */
extern novaStatus_t printHipDeviceInfo(bool verbose);

#endif /* NOVA_HAS_HIP */

/**
 * @var atomic_device_detection_done
 * @brief @c true after the first successful call to a detection
 *        function ( @ref is_device_available, @ref is_cuda_available,
 *        or @ref is_hip_available).
 *
 * @details
 * Once set to @c true, all subsequent detection calls return
 * immediately from the cache without querying the runtime API.
 * Protected by @ref runtime_flags_mtx for thread safety.
 */
static _Atomic bool atomic_device_detection_done = false;

/**
 * @var atomic_detected_device_kind
 * @brief The @ref DeviceKind detected during the first probe.
 *
 * @details
 * Holds @c CUDA_DEVICE, @c HIP_DEVICE, or @c NULL_DEVICE (if no GPU
 * was found).  Written once under @ref runtime_flags_mtx and
 * read thereafter via acquire loads.
 */
static _Atomic DeviceKind atomic_detected_device_kind = NULL_DEVICE;

#ifdef __linux__
/**
 * @var runtime_flags_once
 * @brief C11 @c call_once guard so @ref runtime_flags_mtx is
 *        initialized exactly once, even under concurrent access.
 */
static once_flag runtime_flags_once = {};
#elifdef _WIN64
/**
 * @var runtime_flags_once
 * @brief Windows one-time initialization guard so
 *        @ref runtime_flags_mtx is initialized exactly once.
 */
static INIT_ONCE runtime_flags_once = INIT_ONCE_STATIC_INIT;
#endif

/**
 * @var runtime_flags_mtx
 * @brief Mutex protecting writes to @ref atomic_device_detection_done and
 *        @ref atomic_detected_device_kind.
 *
 * @details
 * Portable @c MTX_T storage (@c mtx_t on Linux, @c CRITICAL_SECTION on
 * Windows), initialized lazily via @ref init_runtime_flags_lock() through
 * the platform one-shot guard. Only taken for write operations; reads are
 * safe without locking due to the single-writer pattern. Use
 * @c MTX_LOCK/@c MTX_UNLOCK to access it.
 */
static MTX_T runtime_flags_mtx;

#ifdef __linux__
/**
 * @brief Lazy initializer for @ref runtime_flags_mtx.
 *
 * @details
 * Passed to C11 @c call_once so that the mutex is created exactly
 * once, regardless of how many threads race to call detection
 * functions.  The return value of @c mtx_init is discarded
 * (@c (void) cast) because the C11 mutex API does not provide a
 * meaningful error path for @c mtx_plain.
 */
static void init_runtime_flags_lock() {
  (void)mtx_init(&runtime_flags_mtx, mtx_plain);
}
#elifdef _WIN64
/**
 * @brief Lazy initializer for @ref runtime_flags_mtx.
 *
 * @details
 * Passed to Windows @c InitOnceExecuteOnce so that the critical
 * section is created exactly once, regardless of how many threads
 * race to call detection functions.
 */
static BOOL CALLBACK init_runtime_flags_lock(PINIT_ONCE once, PVOID param,
                                             PVOID *ctx) {
  (void)once;
  (void)param;
  (void)ctx;
  InitializeCriticalSection(&runtime_flags_mtx);
  return TRUE;
}
#endif

/**
 * @var transf_dispatch
 * @brief Lookup table mapping (src, dst) device pairs to a transfer
 *        kind.
 *
 * @details
 * 3x3 matrix indexed by @ref Device_ (@c DEVICE_CPU=0,
 * @c DEVICE_GPU=1, @c DEVICE_META=2).  Initialized by
 * @ref init_transf_dispatch() at program startup and read-only
 * afterwards, so it is safe to access from any thread.
 *
 * Only three entries are populated: CPU to GPU, GPU to CPU and GPU
 * to GPU.  Every other entry stays zero and is reported as
 * @ref novaInvalidTransfDirection by @ref transfer_to().
 *
 * @see init_transf_dispatch()
 * @see transfer_to()
 * @see TransferKind
 */
static TransferKind transf_dispatch[3][3] = {};

/**
 * @brief Populate the transfer dispatch table at program startup.
 *
 * @details
 * Runs once through @c INITIALIZE before any thread can call
 * @ref transfer_to(), so no locking is needed.  Entries involving
 * @c DEVICE_META and host-to-host stay zero: those pairs are
 * meaningless here (host-to-host copies belong to @c memcpy(), META
 * has no backing storage).
 *
 * @see transf_dispatch
 * @see transfer_to()
 */
INITIALIZE(init_transf_dispatch) {

  transf_dispatch[DEVICE_GPU][DEVICE_CPU] = deviceMemcpyDeviceToHost;
  transf_dispatch[DEVICE_CPU][DEVICE_GPU] = deviceMemcpyHostToDevice;
  transf_dispatch[DEVICE_GPU][DEVICE_GPU] = deviceMemcpyDeviceToDevice;
}

/**
 * @brief Check whether any GPU device backend is available.
 *
 * @details
 * Dispatches to the backend probe for @p kind (@c CUDA_DEVICE to
 * @c isCudaDeviceAvailable(), @c HIP_DEVICE to
 * @c isHipDeviceAvailable(), anything else to @c false).  When the
 * backend macro is not defined the probe compiles to @c false.
 * The first successful probe populates the one-shot cache consumed
 * by every later call.
 *
 * @param[in] kind     Requested backend kind.  Must be a valid
 *                     @ref DeviceKind value.
 * @param[in] verbose  If @c true, backend probes may print runtime
 *                     diagnostics to @c stdout.  Pass @c false for
 *                     silent operation.
 *
 * @return @c true when the requested backend reports an available
 *         device.  @c false otherwise.
 *
 * @see is_cuda_available()          Convenience wrapper for @c CUDA_DEVICE.
 * @see is_hip_available()           Convenience wrapper for @c HIP_DEVICE.
 * @see get_detected_device_kind()   Returns the cached backend.
 * @see was_device_detection_done()  Checks if detection ran.
 */
bool is_device_available(DeviceKind kind, bool verbose) {
  if (atomic_load_explicit(&atomic_device_detection_done,
                           memory_order_acquire) &&
      atomic_load_explicit(&atomic_detected_device_kind,
                           memory_order_acquire) != NULL_DEVICE) {
    return (kind == atomic_load_explicit(&atomic_detected_device_kind,
                                         memory_order_acquire));
  }
#ifdef __linux__
  call_once(&runtime_flags_once, init_runtime_flags_lock);
#elifdef _WIN64
  InitOnceExecuteOnce(&runtime_flags_once, init_runtime_flags_lock, nullptr,
                      nullptr);
#endif

  switch (kind) {
  case CUDA_DEVICE: {
#ifdef NOVA_HAS_CUDA
    if (isCudaDeviceAvailable(verbose, verbose)) {
      MTX_LOCK(&runtime_flags_mtx);
      atomic_store_explicit(&atomic_device_detection_done, true,
                            memory_order_release);
      atomic_store_explicit(&atomic_detected_device_kind, CUDA_DEVICE,
                            memory_order_release);
      MTX_UNLOCK(&runtime_flags_mtx);
      return true;
    }
#endif
    return false;
  }
  case HIP_DEVICE: {
#ifdef NOVA_HAS_HIP
    if (isHipDeviceAvailable(false, false)) {
      MTX_LOCK(&runtime_flags_mtx);
      atomic_store_explicit(&atomic_device_detection_done, true,
                            memory_order_release);
      atomic_store_explicit(&atomic_detected_device_kind, HIP_DEVICE,
                            memory_order_release);
      MTX_UNLOCK(&runtime_flags_mtx);
      return true;
    }
#endif
    return false;
  }
  case NULL_DEVICE:
  default:
    return false;
  }
}

/**
 * @brief Check whether CUDA should be selected as the active backend.
 *
 * @details
 * Equivalent to @c is_device_available(CUDA_DEVICE, false).  Goes
 * through the same one-shot cache.
 *
 * @return @c true when CUDA reports an available device.  @c false if
 *         CUDA is unavailable or @c NOVA_HAS_CUDA is not defined.
 *
 * @see is_hip_available()
 * @see is_device_available()
 */
bool is_cuda_available() {

  if (atomic_load_explicit(&atomic_device_detection_done,
                           memory_order_acquire) &&
      atomic_load_explicit(&atomic_detected_device_kind,
                           memory_order_acquire) != NULL_DEVICE) {
    return (CUDA_DEVICE == atomic_load_explicit(&atomic_detected_device_kind,
                                                memory_order_acquire));
  }

#ifdef __linux__
  call_once(&runtime_flags_once, init_runtime_flags_lock);
#elifdef _WIN64
  InitOnceExecuteOnce(&runtime_flags_once, init_runtime_flags_lock, nullptr,
                      nullptr);
#endif

#ifdef NOVA_HAS_CUDA
  if (isCudaDeviceAvailable(false, false)) {
    MTX_LOCK(&runtime_flags_mtx);
    atomic_store_explicit(&atomic_device_detection_done, true,
                          memory_order_release);
    atomic_store_explicit(&atomic_detected_device_kind, CUDA_DEVICE,
                          memory_order_release);
    MTX_UNLOCK(&runtime_flags_mtx);
    return true;
  }
#endif
  return false;
}

/**
 * @brief Check whether HIP should be selected as the active backend.
 *
 * @details
 * Equivalent to @c is_device_available(HIP_DEVICE, false).  Goes
 * through the same one-shot cache.
 *
 * @return @c true when HIP reports an available device.  @c false if
 *         HIP is unavailable or @c NOVA_HAS_HIP is not defined.
 *
 * @see is_cuda_available()
 * @see is_device_available()
 */
bool is_hip_available() {

  if (atomic_load_explicit(&atomic_device_detection_done,
                           memory_order_acquire) &&
      atomic_load_explicit(&atomic_detected_device_kind,
                           memory_order_acquire) != NULL_DEVICE) {
    return (HIP_DEVICE == atomic_load_explicit(&atomic_detected_device_kind,
                                               memory_order_acquire));
  }

#ifdef __linux__
  call_once(&runtime_flags_once, init_runtime_flags_lock);
#elifdef _WIN64
  InitOnceExecuteOnce(&runtime_flags_once, init_runtime_flags_lock, nullptr,
                      nullptr);
#endif

#ifdef NOVA_HAS_HIP
  if (isHipDeviceAvailable(false, false)) {
    MTX_LOCK(&runtime_flags_mtx);
    atomic_store_explicit(&atomic_device_detection_done, true,
                          memory_order_release);
    atomic_store_explicit(&atomic_detected_device_kind, HIP_DEVICE,
                          memory_order_release);
    MTX_UNLOCK(&runtime_flags_mtx);
    return true;
  }
#endif
  return false;
}

/**
 * @brief Transfer memory between device backends.
 *
 * @details
 * Looks up the @ref TransferKind from @ref transf_dispatch with the
 * (src, dst) pair as indices, then forwards the request to
 * @c deviceTransfer() (declared in @ref ffi.hpp).
 *
 * @param[in]  src      Source device placement.
 * @param[in]  dst      Target device placement.
 * @param[in]  src_buf  Pointer to the source buffer, valid for at
 *                      least @p bytes in the source memory space.
 * @param[out] dst_buf  Pointer to the destination buffer, valid for
 *                      at least @p bytes in the destination memory
 *                      space.
 * @param[in]  bytes    Number of bytes to transfer.  Must be > 0.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, or an error status with a descriptive @c message on
 *         failure.
 *
 * @pre  Both buffers must point to memory regions of at least
 *       @p bytes in their respective address spaces.
 * @pre  @p src and @p dst must be @c DEVICE_CPU or @c DEVICE_GPU.
 *       Host-to-host pairs are rejected with
 *       @ref novaInvalidTransfDirection; use @c memcpy() for those.
 *
 * @see deviceTransfer()  Low-level C-callable copy wrapper.
 * @see transf_dispatch   Lookup table mapping device pairs to
 *                        transfer directions.
 */
novaStatus_t transfer_to(Device_ src, Device_ dst, const void *src_buf,
                         void *dst_buf, size_t bytes) {
  if (src == DEVICE_CPU && dst == DEVICE_CPU) {
    novaStatus_t status;
    status.err = novaInvalidTransfDirection;
    status.message = "Cannot transfer data between host and host; use "
                     "memcpy() instead\n";
    return status;
  }
  TransferKind kind = transf_dispatch[src][dst];
  return deviceTransfer(src_buf, dst_buf, kind, bytes);
}

/**
 * @brief Return the active device id (CUDA or HIP).
 *
 * @details
 * Probes CUDA first, then HIP, so on systems with both runtimes
 * available CUDA takes priority.  The returned id is a 0-based index
 * into the detected runtime's device list; the detection layer
 * currently only supports the first device (id @c 0).
 *
 * @return Active device id (0-based), or @c -1 when no GPU device is
 *         available or detection has not yet been performed.
 *
 * @see is_cuda_available()
 * @see is_hip_available()
 * @see getCudaDeviceId()  CUDA-specific device id accessor.
 * @see getHipDeviceId()   HIP-specific device id accessor.
 */
int get_device_id() {

  if (is_cuda_available()) {
#ifdef NOVA_HAS_CUDA
    return getCudaDeviceId();
#endif
  }
  if (is_hip_available()) {
#ifdef NOVA_HAS_HIP
    return getHipDeviceId();
#endif
  }
  return -1;
}

/**
 * @brief Print detailed or concise device information to stdout.
 *
 * @details
 * Delegates to @c printCudaDeviceInfo() or @c printHipDeviceInfo()
 * depending on @p kind.  With @c NULL_DEVICE (or an unknown kind) it
 * returns @ref novaDeviceNotAvailable, and after an exhausted switch
 * @ref novaBackendNotSupported.
 *
 * @param[in] kind     Backend to query.  Must be @c CUDA_DEVICE or
 *                     @c HIP_DEVICE.  @c NULL_DEVICE is a no-op.
 * @param[in] verbose  If @c true, print the detailed block.  If
 *                     @c false, print the concise summary.
 *
 * @see is_device_available()
 * @see printCudaDeviceInfo()  CUDA backend print function.
 * @see printHipDeviceInfo()   HIP backend print function.
 */
novaStatus_t print_device_info(DeviceKind kind, bool verbose) {
  novaStatus_t status;
  switch (kind) {
  case CUDA_DEVICE: {
#ifdef NOVA_HAS_CUDA
    return printCudaDeviceInfo(verbose);
#endif
    break;
  }
  case HIP_DEVICE: {
#ifdef NOVA_HAS_HIP
    return printHipDeviceInfo(verbose);
#endif
    break;
  }
  case NULL_DEVICE:
  default:
    status.err = novaDeviceNotAvailable;
    status.message = nova_get_error_msg(status.err, nullptr);
    break;
  }
  status.err = novaBackendNotSupported;
  status.message = nova_get_error_msg(status.err, nullptr);
  return status;
}

/**
 * @brief Return the cached device kind from the last detection.
 *
 * @details
 * Returns the @ref DeviceKind value stored by the first call to
 * @ref is_device_available(), @ref is_cuda_available(), or
 * @ref is_hip_available().  If detection has not been performed
 * yet, returns @c NULL_DEVICE.
 *
 * @return The detected @ref DeviceKind, or @c NULL_DEVICE if no
 *         detection has occurred.
 *
 * @see was_device_detection_done()
 * @see is_device_available()
 */
DeviceKind get_detected_device_kind() {
  return atomic_load_explicit(&atomic_detected_device_kind,
                              memory_order_acquire);
}

/**
 * @brief Check whether device detection has already been performed.
 *
 * @details
 * Returns @c true only when the first detection call found a device
 * (i.e., @ref atomic_device_detection_done is set).  A probe that finds no
 * device leaves the flag unset.  Useful for guarding one-time
 * initialization that depends on a detected backend.
 *
 * @return @c true if a device was found by a detection call,
 *         @c false otherwise.
 *
 * @see get_detected_device_kind()
 * @see is_device_available()
 */
bool was_device_detection_done() {
  return atomic_load_explicit(&atomic_device_detection_done,
                              memory_order_acquire);
}
