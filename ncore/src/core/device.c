/**
 * @file device.c
 * @brief Device-backend detection, dispatch-table setup, and
 *        inter-backend memory transfers for the core C layer.
 *
 * @details
 * This translation unit implements the public device API declared in
 * @ref device.h.  It is the glue layer that:
 *
 * 1. **Delegates detection** to backend-specific modules
 *    (@ref cuda_device.c and @ref hip_device.c) via `extern`
 *    declarations.
 * 2. **Initialises the transfer dispatch table** (@ref transf_dispatch)
 *    at program startup using a `__attribute__((constructor))`
 *    function.
 * 3. **Routes memory transfers** through @ref transfer_to(), which
 *    resolves the copy direction from the dispatch table and forwards
 *    the request to the C-callable `device_memcpy_c()` wrapper.
 * 4. **Aggregates backend queries** through @ref is_device_available(),
 *    @ref is_cuda_available(), @ref is_hip_available(), and
 *    @ref get_device_id().
 * 5. **Exposes device information** through @ref print_device_info(),
 *    which delegates to backend-specific print functions.
 *
 * ## Architecture
 *
 * The design follows a **backend-agnostic dispatch** pattern:
 * - Application code calls @ref transfer_to() with abstract `Device`
 *   values.
 * - The dispatch table translates `(src, dst)` pairs into a
 *   `TransferKind` that the underlying runtime understands.
 * - The C-callable `device_memcpy_c()` wrapper (declared in
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
 *   detection functions.
 * - @ref get_device_id() delegates to thread-safe backend accessors.
 * - @ref print_device_info() delegates to thread-safe backend print
 *   functions.
 *
 * @see device.h       Public API declarations.
 * @see cuda_device.c  CUDA detection implementation.
 * @see hip_device.c   HIP detection implementation.
 * @see cpp_ffi.h      C-callable device_memcpy_c() wrapper.
 */

#include <ncore/cpp_ffi.h>
#include <ncore/device.h>
#include <ncore/macros.h>

#ifdef NOVA_HAS_HIP

/**
 * @brief Return the active HIP device id.
 *
 * @details
 * Declared as `extern` because the function is defined in
 * @ref hip_device.c.  This avoids pulling the entire HIP detection
 * module into a single translation unit and keeps the linker
 * responsible for resolving the symbol.
 *
 * @return Device id (0-based), or `-1` when HIP is unavailable.
 *
 * @see get_hip_device_id()  Definition in hip_device.c.
 * @see get_device_id()      Public aggregator that calls this.
 */
extern int get_hip_device_id(void);

/**
 * @brief Probe HIP runtime availability.
 *
 * @details
 * Declared as `extern` because the function is defined in
 * @ref hip_device.c.  The `log` parameter controls whether runtime
 * errors are printed to `stdout`.  The `verbose` parameter controls
 * whether a detailed or concise device information block is printed
 * on success.
 *
 * @param[in] log      If `true`, print HIP runtime error details.
 * @param[in] verbose  If `true`, print detailed device info.
 *
 * @return `true` when a HIP device is available.
 *
 * @see is_hip_device_available()  Definition in hip_device.c.
 * @see is_device_available()      Public dispatcher that calls this.
 */
extern bool is_hip_device_available(bool log, bool verbose);

/**
 * @brief Print HIP device 0 information to stdout.
 *
 * @details
 * Declared as `extern` because the function is defined in
 * @ref hip_device.c.  Called by @ref print_device_info() when the
 * backend is `HIP_DEVICE`.
 *
 * @param[in] verbose  If `true`, print detailed block.  If `false`,
 *                     print concise summary.
 */
extern void print_hip_device_info(bool verbose);
#endif /* NOVA_HAS_HIP */

#ifdef NOVA_HAS_CUDA

/**
 * @brief Return the active CUDA device id.
 *
 * @details
 * Declared as `extern` because the function is defined in
 * @ref cuda_device.c.  This keeps the CUDA detection code in its
 * own translation unit.
 *
 * @return Device id (0-based), or `-1` when CUDA is unavailable.
 *
 * @see get_cuda_device_id()  Definition in cuda_device.c.
 * @see get_device_id()       Public aggregator that calls this.
 */
extern int get_cuda_device_id(void);

/**
 * @brief Probe CUDA runtime availability.
 *
 * @details
 * Declared as `extern` because the function is defined in
 * @ref cuda_device.c.  The `log` parameter controls whether runtime
 * errors are printed to `stdout`.  The `verbose` parameter controls
 * whether a detailed or concise device information block is printed
 * on success.
 *
 * @param[in] log      If `true`, print CUDA runtime error details.
 * @param[in] verbose  If `true`, print detailed device info.
 *
 * @return `true` when a CUDA device is available.
 *
 * @see is_cuda_device_available()  Definition in cuda_device.c.
 * @see is_device_available()       Public dispatcher that calls this.
 */
extern bool is_cuda_device_available(bool log, bool verbose);

/**
 * @brief Print CUDA device 0 information to stdout.
 *
 * @details
 * Declared as `extern` because the function is defined in
 * @ref cuda_device.c.  Called by @ref print_device_info() when the
 * backend is `CUDA_DEVICE`.
 *
 * @param[in] verbose  If `true`, print detailed block.  If `false`,
 *                     print concise summary.
 */
extern void print_cuda_device_info(bool verbose);
#endif /* NOVA_HAS_CUDA */

/**
 * @var transf_dispatch
 * @brief Lookup table to dispatch the type of transfer according to
 *        the source and destination device types.
 *
 * @details
 * A 3×3 matrix indexed by `[dst][src]` where each index is a
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
 * | `[dst]`      | `[src]`      | Value                             |
 * |--------------|--------------|-----------------------------------|
 * | `DEVICE_GPU` | `DEVICE_CPU` | `deviceMemcpyHostToDevice`        |
 * | `DEVICE_CPU` | `DEVICE_GPU` | `deviceMemcpyDeviceToHost`        |
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
ATTR(constructor) static inline void init_transf_dispatch() {

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
 * - `CUDA_DEVICE` → `is_cuda_device_available()` (from
 *   @ref cuda_device.c), guarded by `#ifdef NOVA_HAS_CUDA`.
 * - `HIP_DEVICE` → `is_hip_device_available()` (from
 *   @ref hip_device.c), guarded by `#ifdef NOVA_HAS_HIP`.
 * - `NULL_DEVICE` or any other value → returns `false`.
 *
 * When the corresponding backend macro is not defined, the `#else`
 * branch returns `false` without querying the runtime, ensuring that
 * the function always has a well-defined result even on systems
 * without GPU support.
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
 *       functions.
 *
 * @see is_cuda_available()   Convenience wrapper for `CUDA_DEVICE`.
 * @see is_hip_available()    Convenience wrapper for `HIP_DEVICE`.
 * @see DeviceKind            Enum identifying backends.
 */
bool is_device_available(DeviceKind kind, bool verbose) {
  switch (kind) {
  case CUDA_DEVICE: {
#ifdef NOVA_HAS_CUDA
    return is_cuda_device_available(verbose, verbose);
#else
    return false;
#endif
  }
  case HIP_DEVICE: {
#ifdef NOVA_HAS_HIP
    return is_hip_device_available(verbose, verbose);
#else
    return false;
#endif
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
#ifdef NOVA_HAS_CUDA
  return is_cuda_device_available(false, false);
#else
  return false;
#endif
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
#ifdef NOVA_HAS_HIP
  return is_hip_device_available(false, false);
#else
  return false;
#endif
}

/**
 * @brief Transfer memory between device backends.
 *
 * @details
 * High-level memory transfer function that routes the copy through the
 * correct backend at run time.  The function:
 * 1. Looks up the @ref TransferKind from @ref transf_dispatch using
 *    the `(dst, src)` pair as indices (`transf_dispatch[dst][src]`).
 * 2. Forwards the request to `device_memcpy_c()` (declared in
 *    @ref cpp_ffi.h) with the resolved transfer kind.
 *
 * The dispatch table is initialised at program startup by
 * @ref init_transf_dispatch(), so it is always ready when this
 * function is called.
 *
 * @param[in]  dst       Target device placement.  Determines the
 *                       destination memory space.
 * @param[in]  src       Source device placement.  Determines the
 *                       source memory space.
 * @param[in]  src_buf   Pointer to the source buffer.  Must be valid
 *                       for at least @p bytes bytes in the source
 *                       memory space.
 * @param[out] dst_buf   Pointer to the destination buffer.  Must be
 *                       valid for at least @p bytes bytes in the
 *                       destination memory space.
 * @param[in]  is_pinned Whether the host-side buffer is
 *                       pinned/page-locked.  This affects whether the
 *                       runtime uses synchronous or asynchronous
 *                       transfer.
 * @param[in]  bytes     Number of bytes to transfer.  Must be > 0.
 *
 * @return @ref DeviceStatus with `code` 0 on success, or an error
 *         status with a descriptive `message` on failure.
 *
 * @pre  Both @p src_buf and @p dst_buf must point to valid memory
 *       regions of at least @p bytes.
 * @pre  @p bytes must be greater than zero.
 * @pre  The `(dst, src)` pair must have a valid entry in
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
 *       initialisation, and `device_memcpy_c()` is expected to be
 *       thread-safe.
 *
 * @see device_memcpy_c()  Low-level C-callable copy wrapper.
 * @see transf_dispatch    Lookup table mapping device pairs to
 *                         transfer directions.
 * @see TransferKind       Enum encoding copy directions.
 */
DeviceStatus transfer_to(Device dst, Device src, const void *src_buf,
                         void *dst_buf, bool is_pinned, size_t bytes) {
  TransferKind kind = transf_dispatch[dst][src];
  return device_memcpy_c(src_buf, dst_buf, is_pinned, kind, bytes);
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
 * if (is_cuda_available())  return get_cuda_device_id();
 * if (is_hip_available())   return get_hip_device_id();
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
 * @see get_cuda_device_id()  CUDA-specific device id accessor.
 * @see get_hip_device_id()   HIP-specific device id accessor.
 */
int get_device_id(void) {

  if (is_cuda_available()) {
#ifdef NOVA_HAS_CUDA
    return get_cuda_device_id();
#endif
  }
  if (is_hip_available()) {
#ifdef NOVA_HAS_HIP
    return get_hip_device_id();
#endif
  }
  return -1;
}

/**
 * @brief Print detailed or concise device information to stdout.
 *
 * @details
 * Delegates to the backend-specific print function based on @p kind:
 * - `CUDA_DEVICE` → `print_cuda_device_info()` (from
 *   @ref cuda_device.c), guarded by `#ifdef NOVA_HAS_CUDA`.
 * - `HIP_DEVICE` → `print_hip_device_info()` (from
 *   @ref hip_device.c), guarded by `#ifdef NOVA_HAS_HIP`.
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
 * @see print_cuda_device_info()  CUDA backend print function.
 * @see print_hip_device_info()   HIP backend print function.
 */
void print_device_info(DeviceKind kind, bool verbose) {
  switch (kind) {
  case CUDA_DEVICE: {
#ifdef NOVA_HAS_CUDA
    print_cuda_device_info(verbose);
#endif
    break;
  }
  case HIP_DEVICE: {
#ifdef NOVA_HAS_HIP
    print_hip_device_info(verbose);
#endif
    break;
  }
  case NULL_DEVICE:
  default:
    break;
  }
}
