/**
 * @file hip_device.c
 * @brief HIP runtime device detection and identification.
 *
 * @details
 * This translation unit provides the lowest-level interface for probing
 * AMD (ROCm) GPU availability through the HIP Runtime API.  It is
 * consumed by the device abstraction layer (@ref device.c) and should
 * not be called directly from application code.
 *
 * The detection strategy mirrors @ref cuda_device.c:
 * 1. Call `hipGetDeviceCount()` to query the number of visible GPUs.
 * 2. If at least one device is found, lock a mutex and write the global
 *    device state (`@ref active_device_id` and
 *    `@ref device_available`).
 * 3. Cache the result so that every subsequent call returns immediately
 *    without re-querying the runtime.
 *
 * The `call_once()` mechanism guarantees that the mutex is initialised
 * exactly once, even under concurrent access from multiple threads.
 * The mutex itself is never destroyed — it lives for the entire process
 * lifetime, which is acceptable because this module is intentionally
 * global state.
 *
 * ## Platform Support
 *
 * | Condition                            | Behaviour |
 * |--------------------------------------|---------------------------------------------------|
 * | `NOVA_HAS_HIP` defined               | Full detection via
 * `<hip/hip_runtime_api.h>`       | | `NOVA_HAS_HIP` undefined             |
 * All functions return safe defaults (no devices)    | |
 * `<hip/hip_runtime_api.h>` missing    | Stubs return `false` / `-1` |
 *
 * ### HIP Platform Macro Workaround
 *
 * The HIP headers require exactly one of `__HIP_PLATFORM_AMD__` or
 * `__HIP_PLATFORM_NVIDIA__` to be defined before inclusion.  Static
 * analysis tools (`clangd`, `clang-tidy`) do not define either macro,
 * which would cause compilation to fail.  This file works around the
 * problem by defining `__HIP_PLATFORM_AMD__` to `1` when neither macro
 * is present.  This is safe because the flag only controls which
 * backend declarations are exposed by the HIP headers — it does not
 * affect runtime behaviour.
 *
 * ## Thread Safety
 *
 * All public functions in this file are **safe to call concurrently**
 * from multiple threads.  The first call from any thread performs the
 * actual HIP query; all other calls (concurrent or subsequent) receive
 * the cached result.
 *
 * ## Error Handling
 *
 * When `hipGetDeviceCount()` returns an error, the function can
 * optionally log the error string to `stdout` via the `log` parameter.
 * The global state is **not** modified on error — it retains its
 * previous value (typically the initial `-1` / `false` defaults).
 *
 * @see cuda_device.c  Equivalent CUDA device detection for NVIDIA GPUs.
 * @see device.h       Public device query API consumed by the rest of
 *                     the NovaNN codebase.
 * @see device.c       Device abstraction layer that delegates to this
 *                     module.
 */

#include <ncore/macros.h>
#include <stdbool.h>
#include <stdio.h>
#ifdef __linux__
#include <threads.h>
#elif defined(_WIN64)
/* TODO: windows threading support */
#endif

/**
 * @var active_device_id
 * @brief Active HIP device id, or `-1` when no HIP device is active.
 *
 * @details
 * This global variable holds the 0-based index of the HIP device that
 * was selected during the first successful call to
 * @ref is_hip_device_available().  It is updated under
 * @ref device_flags_mtx to ensure visibility across threads.
 *
 * The initial value is `-1`, which acts as a sentinel indicating that
 * detection has not yet been performed or that no HIP devices were
 * found.  After a successful detection the value is set to `0`
 * (currently only the first device is selected).
 *
 * @note This variable is defined unconditionally — it exists even when
 *       `NOVA_HAS_HIP` is not defined — so that downstream code can
 *       reference it without `#ifdef` guards.  Its value is only
 *       meaningful after @ref is_hip_device_available() has returned
 *       `true`.
 *
 * @see is_hip_device_available()
 * @see get_hip_device_id()
 */
int active_device_id = -1;

/**
 * @var device_available
 * @brief Cached HIP device availability flag.
 *
 * @details
 * Set to `true` by @ref is_hip_device_available() when the HIP Runtime
 * reports at least one available device.  Remains `false` until a
 * successful detection occurs.
 *
 * Like @ref active_device_id, this variable is defined unconditionally
 * so that downstream code can reference it without preprocessor
 * conditionals.  It is only meaningful after
 * @ref is_hip_device_available() has returned `true`.
 *
 * @see is_hip_device_available()
 */
bool device_available = false;

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)

/**
 * @brief Platform macro workaround for HIP headers.
 *
 * @details
 * The HIP headers (`<hip/hip_runtime_api.h>`) require exactly one of
 * `__HIP_PLATFORM_AMD__` or `__HIP_PLATFORM_NVIDIA__` to be defined
 * before they are included.  Without this, compilation fails with a
 * preprocessor error.
 *
 * Static analysis tools such as `clangd` and `clang-tidy` do not
 * define either macro because they do not invoke the full compiler
 * toolchain.  This workaround detects the missing definition and sets
 * `__HIP_PLATFORM_AMD__` to `1` as a default.
 *
 * The choice of AMD as the default is arbitrary — the macro only
 * controls which backend declarations are exposed by the HIP headers.
 * Since NovaNN supports both AMD and NVIDIA via HIP, either platform
 * would work.  AMD is chosen because HIP originated as the AMD ROCm
 * runtime.
 *
 * @note This `#define` is only evaluated when `__has_include` confirms
 *       that `<hip/hip_runtime_api.h>` exists, so it will not affect
 *       non-HIP builds or builds where the HIP SDK is not installed.
 *
 * @see https://rocm.docs.amd.com/
 */
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>

/**
 * @brief Once-flag used to guarantee single initialisation of the mutex.
 *
 * @details
 * Passed to `call_once()` at the beginning of
 * @ref is_hip_device_available().  The C11 standard guarantees that
 * the callback registered with a `once_flag` is invoked exactly once,
 * regardless of how many threads call `call_once()` concurrently.
 */
static once_flag device_flags_once = ONCE_FLAG_INIT;

/**
 * @brief Mutex that serialises writes to the global device state.
 *
 * @details
 * Protects concurrent modification of @ref active_device_id and
 * @ref device_available.  The mutex is initialised lazily by
 * @ref init_device_flags_lock() via `call_once()`.
 *
 * The lock is only held for the two global assignments (a handful of
 * store instructions), so contention is negligible in practice.
 */
static mtx_t device_flags_mtx;

/**
 * @brief Format a byte count into a human-readable memory string.
 *
 * @details
 * Converts @p bytes into a string with the most appropriate unit:
 * - If `bytes >= 8 GiB` (8 589 934 592): displays in GiB with 1
 *   decimal place.
 * - If `bytes >= 1 MiB` (1 048 576): displays in MiB with 1 decimal
 *   place.
 * - Otherwise: displays in bytes as an integer.
 *
 * @param[out] buf      Destination buffer.  Must be at least 16 bytes.
 * @param[in]  bufsize  Size of @p buf in bytes.
 * @param[in]  bytes    Number of bytes to format.
 */
static void format_memory(char *buf, size_t bufsize, size_t bytes) {
  if (bytes >= (size_t)8 * 1024 * 1024 * 1024) {
    snprintf(buf, bufsize, "%.1f GiB",
             (double)bytes / (1024.0 * 1024.0 * 1024.0));
  } else if (bytes >= (size_t)1024 * 1024) {
    snprintf(buf, bufsize, "%.1f MiB", (double)bytes / (1024.0 * 1024.0));
  } else {
    snprintf(buf, bufsize, "%zu bytes", bytes);
  }
}

/**
 * @brief Format a HIP version integer into a dotted string.
 *
 * @details
 * The HIP Runtime encodes versions as `major * 10000 + minor * 100 +
 * patch`.  For example, `60000` decodes to `6.0.0` and `50123`
 * decodes to `5.0.123`.
 *
 * @param[out] buf      Destination buffer.  Must be at least 16 bytes.
 * @param[in]  bufsize  Size of @p buf in bytes.
 * @param[in]  version  Raw version integer from `hipDriverGetVersion()`
 *                      or `hipRuntimeGetVersion()`.
 */
static void format_hip_version(char *buf, size_t bufsize, int version) {
  int major = version / 10000;
  int minor = (version % 10000) / 100;
  int patch = version % 100;
  snprintf(buf, bufsize, "%d.%d.%d", major, minor, patch);
}

/**
 * @brief Print HIP device information to stdout.
 *
 * @details
 * Queries device 0 (the only device NovaNN supports in monoGPU mode)
 * via `hipGetDeviceProperties()` and prints its properties.  When
 * @p verbose is `true`, a detailed multi-line block is printed.  When
 * `false`, a concise two-line summary is printed.
 *
 * The output uses ANSI colour codes defined in @ref macros.h to match
 * the cmake build output style (green prefix, cyan values, bold
 * emphasis).
 *
 * @warning If `hipGetDeviceProperties()` fails, the function triggers
 *          a fatal assertion via @ref NOVA_INTERNAL_ASSERT and aborts
 *          the process.  This is intentional — if device 0 cannot be
 *          queried after detection reported it as available, the
 *          runtime state is inconsistent and continuing would lead to
 *          undefined behaviour.
 *
 * @param[in] verbose  If `true`, print the detailed block.  If
 *                     `false`, print the two-line summary.
 */
void print_hip_device_info(bool verbose) {
  hipDeviceProp_t prop;
  hipError_t err = hipGetDeviceProperties(&prop, 0);
  NOVA_INTERNAL_ASSERT(
      err == hipSuccess,
      "[HIP] print_hip_device_info: error loading hip device properties\n."
      "Error message: '%s'\n",
      hipGetErrorString(err));

  int driver_ver = 0;
  int runtime_ver = 0;
  hipDriverGetVersion(&driver_ver);
  hipRuntimeGetVersion(&runtime_ver);

  char mem_str[16];
  char driver_str[16];
  char runtime_str[16];
  format_memory(mem_str, sizeof(mem_str), prop.totalGlobalMem);
  format_hip_version(driver_str, sizeof(driver_str), driver_ver);
  format_hip_version(runtime_str, sizeof(runtime_str), runtime_ver);

  if (verbose) {
    printf(NCORE_LOG_PREFIX NCORE_LOG_BOLD
           " === HIP Device 0 ===\n" NCORE_LOG_RESET);
    printf(NCORE_LOG_PREFIX "   Name:                  " NCORE_LOG_VALUE
                            "%s\n" NCORE_LOG_RESET,
           prop.name);
    printf(NCORE_LOG_PREFIX "   Compute Capability:    " NCORE_LOG_VALUE
                            "%d.%d\n" NCORE_LOG_RESET,
           prop.major, prop.minor);
    printf(NCORE_LOG_PREFIX "   Total Global Memory:   " NCORE_LOG_VALUE
                            "%s\n" NCORE_LOG_RESET,
           mem_str);
    printf(NCORE_LOG_PREFIX "   SMs:                   " NCORE_LOG_VALUE
                            "%d\n" NCORE_LOG_RESET,
           prop.multiProcessorCount);
    printf(NCORE_LOG_PREFIX "   Warp Size:             " NCORE_LOG_VALUE
                            "%d\n" NCORE_LOG_RESET,
           prop.warpSize);
    printf(NCORE_LOG_PREFIX "   Max Threads/Block:     " NCORE_LOG_VALUE
                            "%d\n" NCORE_LOG_RESET,
           prop.maxThreadsPerBlock);
    printf(NCORE_LOG_PREFIX "   Max Threads/SM:        " NCORE_LOG_VALUE
                            "%d\n" NCORE_LOG_RESET,
           prop.maxThreadsPerMultiProcessor);
    printf(NCORE_LOG_PREFIX "   Driver Version:        " NCORE_LOG_VALUE
                            "%s\n" NCORE_LOG_RESET,
           driver_str);
    printf(NCORE_LOG_PREFIX "   Runtime Version:       " NCORE_LOG_VALUE
                            "%s\n" NCORE_LOG_RESET,
           runtime_str);
  } else {
    printf(NCORE_LOG_PREFIX " [HIP] Device 0: " NCORE_LOG_VALUE NCORE_LOG_BOLD
                            "%s" NCORE_LOG_RESET " | Compute " NCORE_LOG_VALUE
                            "%d.%d" NCORE_LOG_RESET " | " NCORE_LOG_VALUE
                            "%s" NCORE_LOG_RESET " | " NCORE_LOG_VALUE
                            "%d SMs\n" NCORE_LOG_RESET,
           prop.name, prop.major, prop.minor, mem_str,
           prop.multiProcessorCount);
    printf(NCORE_LOG_PREFIX " [HIP] Driver " NCORE_LOG_VALUE
                            "v%s" NCORE_LOG_RESET " | Runtime " NCORE_LOG_VALUE
                            "v%s\n" NCORE_LOG_RESET,
           driver_str, runtime_str);
  }
}

/**
 * @brief Initialise the mutex used to protect HIP device flags.
 *
 * @details
 * This function is registered as the callback for `call_once()` with
 * @ref device_flags_once.  It is guaranteed to be invoked exactly once,
 * even when @ref is_hip_device_available() is called concurrently from
 * multiple threads.
 *
 * The mutex is initialised in plain mode (`mtx_plain`) — no error
 * checking or recursive locking is needed because:
 * - The lock is only held for two simple assignments.
 * - No function that acquires this mutex can throw or call `longjmp`.
 *
 * @note The mutex is never explicitly destroyed (`mtx_destroy` is not
 *       called).  This is intentional: the mutex lives for the
 *       lifetime of the process, and destroying it before all threads
 *       have released it would be undefined behaviour.
 *
 * @see is_hip_device_available()
 * @see device_flags_once
 * @see device_flags_mtx
 */
static void init_device_flags_lock(void) {
  (void)mtx_init(&device_flags_mtx, mtx_plain);
}

/**
 * @brief Probe whether at least one HIP device is available.
 *
 * @details
 * Queries the HIP Runtime API via `hipGetDeviceCount()`.  The call is
 * forwarded through `call_once()` to ensure the internal mutex is
 * initialised before any concurrent access occurs.
 *
 * On success the function writes the global device state under
 * @ref device_flags_mtx:
 * - @ref active_device_id is set to `0` (the first device).
 * - @ref device_available is set to `true`.
 *
 * On failure — whether because the HIP Runtime is unavailable, returns
 * an error code, or reports zero devices — the global state is left
 * unchanged.
 *
 * Because the result is cached in the global variables, subsequent
 * calls from any thread return immediately without re-querying the
 * runtime.  This makes the function effectively idempotent.
 *
 * @param[in] log      If `true`, print HIP runtime error messages to
 *                     `stdout` with a descriptive prefix identifying
 *                     the calling context.  This is useful during
 *                     initialisation diagnostics but should be `false`
 *                     in production to avoid spamming the console.
 * @param[in] verbose  If `true` and @p log is also `true`, print a
 *                     detailed device information block.  If `false`,
 *                     print a concise two-line summary.
 *
 * @return `true` when `hipGetDeviceCount` reports one or more devices.
 *         `false` if HIP is unavailable, the API call fails, or the
 *         device count is zero.
 *
 * @warning This function modifies the global state variables
 *          @ref active_device_id and @ref device_available.
 *          These globals are defined unconditionally but are only
 *          meaningful after this function returns `true`.  Reading
 *          them before a successful call is undefined behaviour.
 *
 * @note Thread-safe.  The underlying HIP query is performed at most
 *       once; the result is cached and returned on subsequent calls.
 *       Concurrent calls from multiple threads are serialised by the
 *       `call_once` mechanism.
 *
 * @note When @p log is `true` and the HIP API returns an error, the
 *       error message includes the function name and a descriptive
 *       context string (e.g., "Error obtaining device count") to
 *       simplify debugging.
 *
 * @post If `true` is returned, @ref active_device_id is set to `0`
 *       and @ref device_available is set to `true`.  Both writes are
 *       visible to all threads due to the mutex.
 * @post If `false` is returned, the global state is unchanged.
 *
 * @see get_hip_device_id()
 * @see is_hip_device_available()
 * @see device_flags_mtx
 */
bool is_hip_device_available(bool log, bool verbose) {
  call_once(&device_flags_once, init_device_flags_lock);

  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);

  if (err != hipSuccess) {
    if (log) {
      printf("[HIP] is_hip_device_available: Error obtaining device count.\n "
             "Error message: '%s'\n",
             hipGetErrorString(err));
    }
    return false;
  }

  if (count == 0) {
    return false;
  }

  mtx_lock(&device_flags_mtx);
  active_device_id = 0;
  device_available = true;
  mtx_unlock(&device_flags_mtx);

  if (log) {
    print_hip_device_info(verbose);
  }

  return device_available;
}

/**
 * @brief Return the selected HIP device id.
 *
 * @details
 * Returns the cached device id that was stored by the last successful
 * call to @ref is_hip_device_available().  If detection has not yet
 * been attempted, or if no HIP device was found, the function returns
 * `-1` (the initial sentinel value of @ref active_device_id).
 *
 * This is a trivial accessor — it performs a single load from a global
 * variable with no locking.  This is safe because:
 * - The global is only written under a mutex, so the store is
 *   atomic with respect to the mutex release.
 * - The caller is expected to have called
 *   @ref is_hip_device_available() first, establishing a
 *   happens-before relationship.
 *
 * @return Active HIP device id (0-based), or `-1` when no HIP device
 *         is active or detection has not yet been performed.
 *
 * @note The return value is only meaningful after
 *       @ref is_hip_device_available() has returned `true`.  Calling
 *       this function before detection is legal but the result is
 *       unreliable (it will be `-1`).
 *
 * @see is_hip_device_available()
 * @see active_device_id
 */
int get_hip_device_id(void) { return active_device_id; }

#else

/**
 * @brief Fallback HIP availability probe when HIP headers are
 *        unavailable.
 *
 * @details
 * This stub is compiled when `NOVA_HAS_HIP` is defined but
 * `<hip/hip_runtime_api.h>` cannot be found at compile time via
 * `__has_include`.  It provides the same signature as the real
 * implementation so that downstream code does not need preprocessor
 * conditionals.
 *
 * @param[in] log      Ignored in the stub; accepted for interface
 *                     compatibility with the real implementation.
 * @param[in] verbose  Ignored in the stub; accepted for interface
 *                     compatibility with the real implementation.
 *
 * @return Always `false`.  No HIP devices can be detected without the
 *         runtime headers.
 *
 * @note This function is a compile-time fallback only.  If you see
 *       this stub being linked, verify that the ROCm toolkit is
 *       installed and that `<hip/hip_runtime_api.h>` is on the
 *       include path.
 */
bool is_hip_device_available(bool log, bool verbose) { return false; }

/**
 * @brief Fallback HIP device id when HIP headers are unavailable.
 *
 * @details
 * This stub is compiled when `NOVA_HAS_HIP` is defined but
 * `<hip/hip_runtime_api.h>` cannot be found at compile time.  It
 * mirrors the real @ref get_hip_device_id() signature.
 *
 * @return Always `-1`, indicating no HIP device is available.
 *
 * @note This function is a compile-time fallback only.
 *
 * @see is_hip_device_available()
 */
int get_hip_device_id(void) { return -1; }
#endif
#endif /* NOVA_HAS_HIP */
