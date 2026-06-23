/**
 * @file device.c
 * @brief Device-backend detection, dispatch-table setup, and
 *        inter-backend memory transfers for the core C layer.
 *
 * @details
 * This translation unit implements the public device API declared in
 * @ref device.h.  It is the glue layer that:
 *
 * 1. **Delegates detection**: to backend-specific modules
 *    (`DetectCudaDevice.cpp` and `DetectHipDevice.cpp`) via native
 *    backend headers.
 * 2. **Initialize transfer dispatch table**: ( @ref transf_dispatch)
 *    at program startup using a `__attribute__((constructor))`
 *    function.
 * 3. **Routes memory transfers**: through @ref transfer_to(), which
 *    resolves the copy direction from the dispatch table and forwards
 *    the request to the C-callable `device_transfer_c()` wrapper.
 * 4. **Aggregates backend queries**: through @ref is_device_available(),
 *    @ref is_cuda_available(), @ref is_hip_available(), and
 *    @ref get_device_id().
 * 5. **Exposes device information**: through @ref print_device_info(),
 *    which delegates to backend-specific print functions.
 *
 * ## Architecture
 *
 * The design follows a **backend-agnostic dispatch** pattern:
 * - Application code calls @ref transfer_to() with abstract `Device`
 *   values.
 * - The dispatch table translates `(src, dst)` pairs into a
 *   `TransferKind` that the underlying runtime understands.
 * - The C-callable `device_transfer_c()` wrapper (declared in
 *   @ref cpp_ffi.h) performs the actual copy using the correct
 *   runtime API.
 *
 * This indirection means that application code never needs to
 * `#include` CUDA or HIP headers directly.
 *
 * ## Thread Safety
 *
 * - @ref transf_dispatch is written once during program startup (by
 *   the constructor) and is read-only thereafter.  All public
 *   functions that read it are safe to call concurrently.
 * - @ref is_device_available() delegates to thread-safe backend
 *   detection functions and caches the result behind a mutex.
 * - @ref get_device_id() delegates to thread-safe backend accessors.
 * - @ref print_device_info() delegates to thread-safe backend print
 *   functions.
 *
 * ## One-shot Caching
 *
 * Device detection is performed once and cached in the module-level
 * variables @ref device_detection_done and @ref detected_device_kind.
 * This avoids repeated runtime API calls during training, where only
 * a single GPU vendor (CUDA _or_ HIP) is ever used.  The cache is
 * protected by a platform-specific mutex and one-shot initialisation
 * guard to ensure thread safety:
 * - **Linux**: C11 `mtx_t` + `call_once` from `<threads.h>`.
 * - **Windows (_WIN64)**: `CRITICAL_SECTION` + `INIT_ONCE` from
 *   `<windows.h>`.
 *
 * @see device.h              Public API declarations.
 * @see DetectCudaDevice.cpp  CUDA detection implementation.
 * @see DetectHipDevice.cpp   HIP detection implementation.
 * @see status.h              novaStatus_t struct.
 * @see cpp_ffi.h             C-callable device_transfer_c() wrapper.
 */

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/ffi/cpp_ffi.h>
#include <ncore/headeronly/macros.h>
#include <stdbool.h>

#ifdef __linux__
#include <threads.h>
#elif defined(_WIN64)
#include <windows.h>
#endif

#ifdef NOVA_HAS_CUDA

/**
 * @brief Probe the CUDA runtime for an available GPU device.
 *
 * @details
 * Defined in `ncore/native/cuda/DetectCudaDevice.cpp`.  Queries the
 * CUDA runtime API to determine whether a usable GPU is present.
 *
 * @param log      If `true`, print detection results to stdout.
 * @param verbose  If `true`, print detailed device properties.
 *
 * @return `true` when a CUDA-capable device is found.
 *
 * @see is_device_available()
 * @see is_cuda_available()
 */
extern bool isCudaDeviceAvailable(bool log, bool verbose);

/**
 * @brief Return the 0-based device index of the detected CUDA GPU.
 *
 * @details
 * Defined in `ncore/native/cuda/DetectCudaDevice.cpp`.  Returns the
 * device id set during the most recent detection probe.
 *
 * @return 0-based device index, or `-1` if no CUDA device was detected.
 *
 * @see get_device_id()
 */
extern int getCudaDeviceId();

/**
 * @brief Print detailed CUDA device properties to stdout.
 *
 * @details
 * Defined in `ncore/native/cuda/DetectCudaDeviceInfo.cpp`.  Delegates
 * to the CUDA runtime to query and display device capabilities.
 *
 * @param verbose  If `true`, print the full property block.  If
 *                 `false`, print a concise summary.
 *
 * @return @ref novaStatus_t with the result of the detection.
 *
 * @see print_device_info()
 */
extern novaStatus_t printCudaDeviceInfo(bool verbose);

#endif /* NOVA_HAS_CUDA */

#ifdef NOVA_HAS_HIP

/**
 * @brief Probe the HIP runtime for an available GPU device.
 *
 * @details
 * Defined in `ncore/native/hip/DetectHipDevice.cpp`.  Queries the
 * HIP runtime API to determine whether a usable GPU is present.
 *
 * @param log      If `true`, print detection results to stdout.
 * @param verbose  If `true`, print detailed device properties.
 *
 * @return `true` when a HIP-capable device is found.
 *
 * @see is_device_available()
 * @see is_hip_available()
 */
extern bool isHipDeviceAvailable(bool log, bool verbose);

/**
 * @brief Return the 0-based device index of the detected HIP GPU.
 *
 * @details
 * Defined in `ncore/native/hip/DetectHipDevice.cpp`.  Returns the
 * device id set during the most recent detection probe.
 *
 * @return 0-based device index, or `-1` if no HIP device was detected.
 *
 * @see get_device_id()
 */
extern int getHipDeviceId();

/**
 * @brief Print detailed HIP device properties to stdout.
 *
 * @details
 * Defined in `ncore/native/hip/DetectHipDeviceInfo.cpp`.  Delegates
 * to the HIP runtime to query and display device capabilities.
 *
 * @param verbose  If `true`, print the full property block.  If
 *                 `false`, print a concise summary.
 *
 * @return @ref novaStatus_t with the result of the detection.
 *
 * @see print_device_info()
 */
extern novaStatus_t printHipDeviceInfo(bool verbose);

#endif /* NOVA_HAS_HIP */

/**
 * @var device_detection_done
 * @brief `true` after the first successful call to a detection
 *        function ( @ref is_device_available, @ref is_cuda_available,
 *        or @ref is_hip_available).
 *
 * @details
 * Once set to `true`, all subsequent detection calls return
 * immediately from the cache without querying the runtime API.
 * Protected by @ref runtime_flags_mtx for thread safety.
 */
static bool device_detection_done = false;

/**
 * @var detected_device_kind
 * @brief The @ref DeviceKind detected during the first probe.
 *
 * @details
 * Holds `CUDA_DEVICE`, `HIP_DEVICE`, or `NULL_DEVICE` (if no GPU
 * was found).  Written once under @ref runtime_flags_mtx and
 * read thereafter without locking (single-writer guarantee).
 */
static DeviceKind detected_device_kind = NULL_DEVICE;

#ifdef __linux__
/**
 * @var runtime_flags_once
 * @brief C11 `call_once` guard ensuring @ref runtime_flags_mtx is
 *        initialised exactly once, even under concurrent access.
 */
static once_flag runtime_flags_once = ONCE_FLAG_INIT;
#elif defined(_WIN64)
/**
 * @var runtime_flags_once
 * @brief Windows one-time initialisation guard ensuring
 *        @ref runtime_flags_mtx is initialised exactly once.
 */
static INIT_ONCE runtime_flags_once = INIT_ONCE_STATIC_INIT;
#endif

#ifdef __linux__
/**
 * @var runtime_flags_mtx
 * @brief Mutex protecting writes to @ref device_detection_done and
 *        @ref detected_device_kind.
 *
 * @details
 * Initialised lazily via @ref init_runtime_flags_lock() through
 * C11 `call_once`.  Only taken for write operations; reads are
 * safe without locking due to the single-writer pattern.
 */
static mtx_t runtime_flags_mtx;
#elif defined(_WIN64)
/**
 * @var runtime_flags_mtx
 * @brief Critical section protecting writes to
 *        @ref device_detection_done and @ref detected_device_kind.
 *
 * @details
 * Initialised lazily via @ref init_runtime_flags_lock() through
 * Windows `InitOnceExecuteOnce`.  Only taken for write operations;
 * reads are safe without locking due to the single-writer pattern.
 */
static CRITICAL_SECTION runtime_flags_mtx;
#endif

#ifdef __linux__
/**
 * @brief Lazy initialiser for @ref runtime_flags_mtx.
 *
 * @details
 * Passed to C11 `call_once` so that the mutex is created exactly
 * once, regardless of how many threads race to call detection
 * functions.  The return value of `mtx_init` is discarded
 * (`(void)` cast) because the C11 mutex API does not provide a
 * meaningful error path for `mtx_plain`.
 */
static void init_runtime_flags_lock(void) {
  (void)mtx_init(&runtime_flags_mtx, mtx_plain);
}
#elif defined(_WIN64)
/**
 * @brief Lazy initialiser for @ref runtime_flags_mtx.
 *
 * @details
 * Passed to Windows `InitOnceExecuteOnce` so that the critical
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
 * @brief Lookup table to dispatch the type of transfer according to
 *        the source and destination device types.
 *
 * @details
 * A 3×3 matrix indexed by `[src][dst]` where each index is a
 * @ref Device value (`DEVICE_CPU=0`, `DEVICE_GPU=1`,
 * `DEVICE_META=2`).  Each entry is a @ref TransferKind value that
 * encodes the correct copy direction for the given `(src, dst)` pair.
 *
 * The table is populated by @ref init_transf_dispatch() at program
 * startup (via `__attribute__((constructor))`).  After initialisation
 * it is read-only and safe to access from any thread.
 *
 * ## Initialised Entries
 *
 * | `[src]`      | `[dst]`      | Value                             |
 * |--------------|--------------|-----------------------------------|
 * | `DEVICE_CPU` | `DEVICE_GPU` | `deviceMemcpyHostToDevice`        |
 * | `DEVICE_GPU` | `DEVICE_CPU` | `deviceMemcpyDeviceToHost`        |
 * | `DEVICE_GPU` | `DEVICE_GPU` | `deviceMemcpyDeviceToDevice`      |
 *
 * All other entries remain `0` (zero-initialised).  Calling
 * @ref transfer_to() with an uninitialised pair (e.g.,
 * `DEVICE_CPU` → `DEVICE_CPU`) is undefined behaviour.
 *
 * @see init_transf_dispatch()
 * @see transfer_to()
 * @see TransferKind
 */
TransferKind transf_dispatch[3][3] = {0};

/**
 * @brief Populate the transfer dispatch table at program startup.
 *
 * @details
 * This function is declared with `__attribute__((constructor))`, so
 * it is called automatically before `main()` executes.  It fills in
 * the three meaningful entries of @ref transf_dispatch:
 *
 * - `GPU → CPU`: `deviceMemcpyDeviceToHost`
 * - `CPU → GPU`: `deviceMemcpyHostToDevice`
 * - `GPU → GPU`: `deviceMemcpyDeviceToDevice`
 *
 * The remaining six entries (involving `DEVICE_CPU → DEVICE_CPU` and
 * any pair with `DEVICE_META`) are left as `0`.  These represent
 * operations that are either meaningless (host-to-host copies should
 * use `memcpy()`) or unsupported (META has no backing storage).
 *
 * @note This function is called exactly once, before any thread can
 *       call @ref transfer_to().  No locking is needed.
 *
 * @see transf_dispatch
 * @see transfer_to()
 */
ATTR(constructor) static inline void init_transf_dispatch(void) {

  transf_dispatch[DEVICE_GPU][DEVICE_CPU] = deviceMemcpyDeviceToHost;
  transf_dispatch[DEVICE_CPU][DEVICE_GPU] = deviceMemcpyHostToDevice;
  transf_dispatch[DEVICE_GPU][DEVICE_GPU] = deviceMemcpyDeviceToDevice;
}

/**
 * @brief Check whether any GPU device backend is available.
 *
 * @details
 * Dispatches to the backend-specific detection function based on
 * @p kind:
 * - `CUDA_DEVICE` → `isCudaDeviceAvailable()` (from
 *   `DetectCudaDevice.cpp`), guarded by `#ifdef NOVA_HAS_CUDA`.
 * - `HIP_DEVICE` → `isHipDeviceAvailable()` (from
 *   `DetectHipDevice.cpp`), guarded by `#ifdef NOVA_HAS_HIP`.
 * - `NULL_DEVICE` or any other value → returns `false`.
 *
 * When the corresponding backend macro is not defined, the `#else`
 * branch returns `false` without querying the runtime, ensuring that
 * the function always has a well-defined result even on systems
 * without GPU support.
 *
 * ## One-shot caching
 *
 * The first call to this function performs the actual runtime probe
 * and stores the result in @ref device_detection_done and
 * @ref detected_device_kind.  Subsequent calls return immediately
 * from the cache.  This eliminates redundant runtime API calls in
 * long-running processes (e.g., training loops).
 *
 * @param[in] kind     Requested backend kind.  Must be a valid
 *                     @ref DeviceKind value.
 * @param[in] verbose  If `true`, backend probes may print runtime
 *                     diagnostics to `stdout`.  Pass `false` for
 *                     silent operation.
 *
 * @return `true` when the requested backend reports an available
 *         device.  `false` otherwise.
 *
 * @note Thread-safe.  Delegates to thread-safe backend detection
 *       functions.  The cache write is protected by
 *       @ref runtime_flags_mtx.
 *
 * @see is_cuda_available()   Convenience wrapper for `CUDA_DEVICE`.
 * @see is_hip_available()    Convenience wrapper for `HIP_DEVICE`.
 * @see get_detected_device_kind()  Returns the cached backend.
 * @see was_device_detection_done()  Checks if detection ran.
 * @see DeviceKind            Enum identifying backends.
 */
bool is_device_available(DeviceKind kind, bool verbose) {
  if (device_detection_done && detected_device_kind != NULL_DEVICE) {
    return (kind == detected_device_kind);
  }
#ifdef __linux__
  call_once(&runtime_flags_once, init_runtime_flags_lock);
#elif defined(_WIN64)
  InitOnceExecuteOnce(&runtime_flags_once, init_runtime_flags_lock, NULL, NULL);
#endif

  switch (kind) {
  case CUDA_DEVICE: {
#ifdef NOVA_HAS_CUDA
    if (isCudaDeviceAvailable(verbose, verbose)) {
#ifdef __linux__
      mtx_lock(&runtime_flags_mtx);
      device_detection_done = true;
      detected_device_kind = CUDA_DEVICE;
      mtx_unlock(&runtime_flags_mtx);
#elif defined(_WIN64)
      EnterCriticalSection(&runtime_flags_mtx);
      device_detection_done = true;
      detected_device_kind = CUDA_DEVICE;
      LeaveCriticalSection(&runtime_flags_mtx);
#endif
      return true;
    }
#endif
    return false;
  }
  case HIP_DEVICE: {
#ifdef NOVA_HAS_HIP
    if (isHipDeviceAvailable(false, false)) {
#ifdef __linux__
      mtx_lock(&runtime_flags_mtx);
      device_detection_done = true;
      detected_device_kind = HIP_DEVICE;
      mtx_unlock(&runtime_flags_mtx);
#elif defined(_WIN64)
      EnterCriticalSection(&runtime_flags_mtx);
      device_detection_done = true;
      detected_device_kind = HIP_DEVICE;
      LeaveCriticalSection(&runtime_flags_mtx);
#endif
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
 * Convenience function that probes the CUDA runtime with verbose
 * output disabled.  Equivalent to:
 * @code{.c}
 * is_device_available(CUDA_DEVICE, false);
 * @endcode
 *
 * This function is provided as a shorthand for code paths that only
 * need a boolean yes/no answer without diagnostics.
 *
 * Respects the one-shot caching mechanism: returns the cached result
 * if detection has already been performed (see
 * @ref is_device_available()).
 *
 * @return `true` when CUDA reports an available device.  `false` if
 *         CUDA is unavailable or `NOVA_HAS_CUDA` is not defined.
 *
 * @note Thread-safe.  Does not print diagnostics (verbose is
 *       hardcoded to `false`).
 *
 * @see is_hip_available()
 * @see is_device_available()
 * @see DeviceKind
 */
bool is_cuda_available(void) {

  if (device_detection_done && detected_device_kind != NULL_DEVICE) {
    return (CUDA_DEVICE == detected_device_kind);
  }

#ifdef __linux__
  call_once(&runtime_flags_once, init_runtime_flags_lock);
#elif defined(_WIN64)
  InitOnceExecuteOnce(&runtime_flags_once, init_runtime_flags_lock, NULL, NULL);
#endif

#ifdef NOVA_HAS_CUDA
  if (isCudaDeviceAvailable(false, false)) {
#ifdef __linux__
    mtx_lock(&runtime_flags_mtx);
    device_detection_done = true;
    detected_device_kind = CUDA_DEVICE;
    mtx_unlock(&runtime_flags_mtx);
#elif defined(_WIN64)
    EnterCriticalSection(&runtime_flags_mtx);
    device_detection_done = true;
    detected_device_kind = CUDA_DEVICE;
    LeaveCriticalSection(&runtime_flags_mtx);
#endif
    return true;
  }
#endif
  return false;
}

/**
 * @brief Check whether HIP should be selected as the active backend.
 *
 * @details
 * Convenience function that probes the HIP runtime with verbose
 * output disabled.  Equivalent to:
 * @code{.c}
 * is_device_available(HIP_DEVICE, false);
 * @endcode
 *
 * This function is provided as a shorthand for code paths that only
 * need a boolean yes/no answer without diagnostics.
 *
 * Respects the one-shot caching mechanism: returns the cached result
 * if detection has already been performed (see
 * @ref is_device_available()).
 *
 * @return `true` when HIP reports an available device.  `false` if
 *         HIP is unavailable or `NOVA_HAS_HIP` is not defined.
 *
 * @note Thread-safe.  Does not print diagnostics (verbose is
 *       hardcoded to `false`).
 *
 * @see is_cuda_available()
 * @see is_device_available()
 * @see DeviceKind
 */
bool is_hip_available(void) {

  if (device_detection_done && detected_device_kind != NULL_DEVICE) {
    return (HIP_DEVICE == detected_device_kind);
  }

#ifdef __linux__
  call_once(&runtime_flags_once, init_runtime_flags_lock);
#elif defined(_WIN64)
  InitOnceExecuteOnce(&runtime_flags_once, init_runtime_flags_lock, NULL, NULL);
#endif

#ifdef NOVA_HAS_HIP
  if (isHipDeviceAvailable(false, false)) {
#ifdef __linux__
    mtx_lock(&runtime_flags_mtx);
    device_detection_done = true;
    detected_device_kind = HIP_DEVICE;
    mtx_unlock(&runtime_flags_mtx);
#elif defined(_WIN64)
    EnterCriticalSection(&runtime_flags_mtx);
    device_detection_done = true;
    detected_device_kind = HIP_DEVICE;
    LeaveCriticalSection(&runtime_flags_mtx);
#endif
    return true;
  }
#endif
  return false;
}

/**
 * @brief Transfer memory between device backends.
 *
 * @details
 * High-level memory transfer function that routes the copy through the
 * correct backend at run time.  The function:
 * 1. Looks up the @ref TransferKind from @ref transf_dispatch using
 *    the `(src, dst)` pair as indices (`transf_dispatch[src][dst]`).
 * 2. Forwards the request to `device_transfer_c()` (declared in
 *    @ref cpp_ffi.h) with the resolved transfer kind.
 *
 * The dispatch table is initialised at program startup by
 * @ref init_transf_dispatch(), so it is always ready when this
 * function is called.
 *
 * @param[in]  src       Source device placement.  Determines the
 *                       source memory space.
 * @param[in]  dst       Target device placement.  Determines the
 *                       destination memory space.
 * @param[in]  src_buf   Pointer to the source buffer.  Must be valid
 *                       for at least @p bytes bytes in the source
 *                       memory space.
 * @param[out] dst_buf   Pointer to the destination buffer.  Must be
 *                       valid for at least @p bytes bytes in the
 *                       destination memory space.
 * @param[in]  bytes     Number of bytes to transfer.  Must be > 0.
 *
 * @return @ref DeviceStatus with `code` 0 on success, or an error
 *         status with a descriptive `message` on failure.
 *
 * @pre  Both @p src_buf and @p dst_buf must point to valid memory
 *       regions of at least @p bytes.
 * @pre  @p bytes must be greater than zero.
 * @pre  The `(src, dst)` pair must have a valid entry in
 *       @ref transf_dispatch (i.e., not a host-to-host or META pair).
 * @post On success, @p dst_buf contains a copy of @p src_buf.
 * @post On failure, the source and destination buffers are unchanged.
 *
 * @warning If @p src and @p dst are both `DEVICE_CPU`, the dispatch
 *          table entry is `0` (uninitialised), which may cause
 *          undefined behaviour.  Use `memcpy()` for host-to-host
 *          copies.
 *
 * @note Thread-safe.  The dispatch table is read-only after
 *       initialisation, and `device_transfer_c()` is expected to be
 *       thread-safe.
 *
 * @see device_transfer_c()  Low-level C-callable copy wrapper.
 * @see transf_dispatch    Lookup table mapping device pairs to
 *                         transfer directions.
 * @see TransferKind       Enum encoding copy directions.
 */
DeviceStatus transfer_to(Device src, Device dst, const void *src_buf,
                         void *dst_buf, size_t bytes) {
  DeviceStatus status;
  if (src == DEVICE_CPU && dst == DEVICE_CPU) {
    status.code = -1;
    status.message = "Cannot transfer data between host and host; use "
                     "deepcopy() or memcpy() instead\n";

    return status;
  }
  TransferKind kind = transf_dispatch[src][dst];
  status = device_transfer_c(src_buf, dst_buf, kind, bytes);
  return status;
}

/**
 * @brief Return the active device id (CUDA or HIP).
 *
 * @details
 * Queries the backend-specific detection modules to determine which
 * GPU runtime is active and returns its device id.  The function
 * checks CUDA first via @ref is_cuda_available(), then HIP via
 * @ref is_hip_available(), and returns `-1` if neither is available.
 *
 * The returned id is a 0-based index into the device list of the
 * active runtime.  Currently only the first device (id `0`) is
 * supported by the detection layer.
 *
 * The function uses a cascade pattern:
 * @code{.c}
 * if (is_cuda_available())  return getCudaDeviceId();
 * if (is_hip_available())   return getHipDeviceId();
 * return -1;
 * @endcode
 *
 * This means that on systems with both CUDA and HIP available, CUDA
 * takes priority.
 *
 * @return Active device id (0-based), or `-1` when no GPU device is
 *         available or detection has not yet been performed.
 *
 * @note The return value is only meaningful after at least one of
 *       @ref is_cuda_available() or @ref is_hip_available() has
 *       returned `true`.
 *
 * @see is_cuda_available()
 * @see is_hip_available()
 * @see getCudaDeviceId()  CUDA-specific device id accessor.
 * @see getHipDeviceId()   HIP-specific device id accessor.
 */
int get_device_id(void) {

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
 * Delegates to the backend-specific print function based on @p kind:
 * - `CUDA_DEVICE` → `printCudaDeviceInfo()` (from
 *   `DetectCudaDevice.cpp`), guarded by `#ifdef NOVA_HAS_CUDA`.
 * - `HIP_DEVICE` → `printHipDeviceInfo()` (from
 *   `DetectHipDevice.cpp`), guarded by `#ifdef NOVA_HAS_HIP`.
 * - `NULL_DEVICE` or any other value → no-op.
 *
 * When the corresponding backend macro is not defined, the function
 * silently returns without printing.
 *
 * @param[in] kind     Backend to query.  Must be `CUDA_DEVICE` or
 *                     `HIP_DEVICE`.  `NULL_DEVICE` is a no-op.
 * @param[in] verbose  If `true`, print the detailed block.  If
 *                     `false`, print the concise summary.
 *
 * @note Thread-safe.  Delegates to thread-safe backend print
 *       functions.  Does not require a prior call to
 *       @ref is_device_available().
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
    status.message = nova_get_error_msg(status.err, NULL);
    break;
  }
  status.err = novaBackendNotSupported;
  status.message = nova_get_error_msg(status.err, NULL);
  return status;
}

/**
 * @brief Return the cached device kind from the last detection.
 *
 * @details
 * Returns the @ref DeviceKind value stored by the first call to
 * @ref is_device_available(), @ref is_cuda_available(), or
 * @ref is_hip_available().  If detection has not been performed
 * yet, returns `NULL_DEVICE`.
 *
 * @return The detected @ref DeviceKind, or `NULL_DEVICE` if no
 *         detection has occurred.
 *
 * @see was_device_detection_done()
 * @see is_device_available()
 */
DeviceKind get_detected_device_kind(void) { return detected_device_kind; }

/**
 * @brief Check whether device detection has already been performed.
 *
 * @details
 * Returns `true` after the first detection call has completed.
 * Useful for guarding one-time initialisation that depends on the
 * detection result.
 *
 * @return `true` if detection has been performed at least once,
 *         `false` otherwise.
 *
 * @see get_detected_device_kind()
 * @see is_device_available()
 */
bool was_device_detection_done(void) { return device_detection_done; }
