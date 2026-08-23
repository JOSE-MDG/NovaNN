/**
 * @file simd.c
 * @brief CPU SIMD capability detection and runtime feature detection
 * implementation.
 *
 * @details
 * This module provides runtime detection of CPU SIMD (Single Instruction,
 * Multiple Data) capabilities using the CPUID instruction. It supports
 * detection of SSE, AVX, AVX-512, FMA, F16C, VNNI, AMX, and AVX10
 * instruction sets on both x86_64 Linux and Windows platforms, giving the
 * rest of NovaNN a single, reliable source of truth for which vectorized
 * kernel variants are safe to dispatch to on the current machine.
 *
 * The detected capabilities are cached in a thread-safe singleton pattern
 * using platform-specific threading primitives (C11 @c call_once on Linux,
 * Windows @c InitOnceExecuteOnce on @c _WIN64) to ensure the detection is
 * performed only once, even when called from multiple threads.
 *
 * The module is split in two layers:
 *
 * @li @ref get_simd_capabilities_from_cpuid() — pure decoding of raw CPUID
 *     registers into a @ref SIMDCapabilities structure. No hardware access;
 *     unit-testable against synthetic snapshots.
 * @li @ref detect_simd_capabilities() — platform-specific reader that fills
 *     a @ref SIMDCpuidSnapshot through the CPUID intrinsics and delegates
 *     the decoding to the pure layer.
 *
 * @section platform-support Platform Support
 * @li Windows (_WIN64): Uses __cpuid() and __cpuidex() from intrin.h;
 *   threading via @c INIT_ONCE + @c InitOnceExecuteOnce from @c <windows.h>.
 * @li Linux/Unix: Uses __get_cpuid() and __cpuid_count() from cpuid.h;
 *   threading via C11 @c once_flag + @c call_once from @c <threads.h>.
 *
 * @see simd.h                     Public interface and @ref SIMDCapabilities structure.
 * @see get_simd_capabilities()    Thread-safe singleton accessor.
 */

#ifdef __linux__
#include <threads.h>
#elif defined(_WIN64)
#include <windows.h>
#endif

#ifdef _WIN64
/** @brief Windows intrinsics for CPUID support. */
#include <intrin.h>
#else
/** @brief GCC/Clang CPUID intrinsic support. */
#include <cpuid.h>
#endif

#include <ncore/core/dtype.h>
#include <ncore/simd/simd.h>

/**
 * @var static SIMDCapabilities simd
 * @brief Global cached CPU capabilities structure.
 *
 * @details
 * Stores the detected SIMD features after the first call to
 * @ref get_simd_capabilities(). Once initialized, this structure is
 * read-only and can be safely accessed from multiple threads.
 *
 * @see get_simd_capabilities()
 * @see detect_simd_capabilities()
 */
static SIMDCapabilities simd = {};

#ifdef __linux__
/**
 * @var static once_flag init_flag
 * @brief Once-flag for thread-safe lazy initialization.
 *
 * @details
 * Ensures @ref init_once() is called exactly once, even when
 * @ref get_simd_capabilities() is called concurrently from multiple threads.
 * Initialized to the C23 empty initializer @c = {}.
 *
 * @see init_once()
 * @see get_simd_capabilities()
 */
static once_flag init_flag = {};
#elif defined(_WIN64)
/**
 * @var static INIT_ONCE init_flag
 * @brief Windows one-time initialisation guard for thread-safe lazy
 *        initialization.
 *
 * @details
 * Ensures the CPU capabilities detection callback is called exactly
 * once, even when @ref get_simd_capabilities() is called concurrently
 * from multiple threads.
 *
 * @see get_simd_capabilities()
 */
static INIT_ONCE init_flag = INIT_ONCE_STATIC_INIT;
#endif

/**
 * @brief Maps raw CPUID registers to SIMD capabilities.
 *
 * @details
 * Pure decoding layer: derives every @ref SIMDCapabilities flag from the
 * registers in @p snapshot without touching the hardware. Isolating this
 * logic from the platform-specific CPUID reads makes it unit-testable
 * against synthetic snapshots of processors that are not physically
 * available to the build host (e.g. AMX or AVX10 silicon).
 *
 * Decoding rules (Intel SDM):
 *
 * @li Leaf 1 ECX bits feed sse4_2_, fma3_, avx_ and f16c_.
 * @li Leaf 7 subleaf 0 feeds avx2_, avx512f_, avx512_dq_, avx512_bw_,
 *     avx512_vl_, avx512_vnni_, amx_bf16_, avx512_fp16_ and amx_int8_.
 * @li Leaf 7 subleaf 1 feeds avx2_vnni_, avx512_bf16_, amx_fp16_ and
 *     avx2_int8_; its EDX bit 19 carries the AVX10 presence flag.
 * @li Leaf 24H EBX[7:0] holds the AVX10 version number; it is consulted
 *     only when the AVX10 presence flag is set, since the leaf is
 *     architecturally undefined otherwise. avx10_1_ is set for a version
 *     of at least 1 and avx10_2_ for a version of at least 2.
 * @li Composite flags: amx_ = OR(amx_bf16_, amx_fp16_, amx_int8_) and
 *     vnni_ = OR(avx512_vnni_, avx2_vnni_).
 *
 * No cross-feature gating is applied: every flag is set exactly when its
 * source bit is set (e.g. avx512_fp16_ does not require avx512f_).
 *
 * @param[in]     snapshot  Raw CPUID registers. Must not be null.
 * @param[in,out] caps      Destination capabilities. Must not be null;
 *                          every flag is overwritten.
 *
 * @pre snapshot must point to a valid SIMDCpuidSnapshot structure.
 * @pre caps must point to a valid SIMDCapabilities structure.
 * @post All capability flags in caps reflect the bits in snapshot; the
 *       amx_ and vnni_ composite flags equal the OR of their constituents.
 *
 * @see get_simd_capabilities()
 */
void get_simd_capabilities_from_cpuid(const SIMDCpuidSnapshot *snapshot,
                                      SIMDCapabilities *caps) {

  /* CPUID leaf 1: SSE4.2, FMA3, AVX, F16C detection */
  const uint32 ecx1 = snapshot->leaf1[2];

  caps->sse4_2_ = ((ecx1 & (1 << 20)) != 0); ///< SSE4.2
  caps->fma3_ = ((ecx1 & (1 << 12)) != 0);   ///< FMA3
  caps->avx_ = ((ecx1 & (1 << 28)) != 0);    ///< AVX
  caps->f16c_ = ((ecx1 & (1 << 29)) != 0);   ///< F16C

  /* CPUID leaf 7, subleaf 0: AVX2, AVX-512, AMX */
  const uint32 ebx70 = snapshot->leaf7_0[1];
  const uint32 ecx70 = snapshot->leaf7_0[2];
  const uint32 edx70 = snapshot->leaf7_0[3];

  caps->avx2_ = ((ebx70 & (1 << 5)) != 0);       ///< AVX2
  caps->avx512f_ = ((ebx70 & (1 << 16)) != 0);   ///< AVX-512 Foundation
  caps->avx512_dq_ = ((ebx70 & (1 << 17)) != 0); ///< AVX-512 Doubleword/Quadword
  caps->avx512_bw_ = ((ebx70 & (1 << 30)) != 0); ///< AVX-512 Byte/Word
  caps->avx512_vl_ =
      ((ebx70 & (1U << 31)) != 0); ///< AVX-512 Vector Length extensions
  caps->avx512_vnni_ = ((ecx70 & (1 << 11)) != 0); ///< AVX-512 VNNI
  caps->amx_bf16_ = ((edx70 & (1 << 22)) != 0);    ///< AMX BF16
  caps->avx512_fp16_ = ((edx70 & (1 << 23)) != 0); ///< AVX-512 FP16
  caps->amx_int8_ = ((edx70 & (1 << 25)) != 0);    ///< AMX INT8

  /* CPUID leaf 7, subleaf 1: AVX2 VNNI, AVX-512 BF16, AMX FP16,
   * AVX10 presence flag */
  const uint32 eax71 = snapshot->leaf7_1[0];
  const uint32 edx71 = snapshot->leaf7_1[3];

  caps->avx2_vnni_ = ((eax71 & (1 << 4)) != 0);   ///< AVX2 VNNI
  caps->avx512_bf16_ = ((eax71 & (1 << 5)) != 0); ///< AVX-512 BF16
  caps->amx_fp16_ = ((eax71 & (1 << 21)) != 0);   ///< AMX FP16
  caps->avx2_int8_ = ((edx71 & (1 << 4)) != 0);   ///< AVX2 INT8

  /**
   * @note CPUID.(EAX=07H,ECX=1H):EDX[19] — Intel AVX10 converged vector ISA
   *       presence flag. Leaf 0x24 is only architecturally valid when this
   *       bit is set; it must not be decoded otherwise.
   */
  const bool avx10_present = ((edx71 & (1 << 19)) != 0);

  /* CPUID leaf 0x24, subleaf 0, EBX[7:0]: AVX10 version number (1 or 2).
   * Only decoded when avx10_present is set, since the leaf is otherwise
   * undefined. EBX[18:16] (legacy per-width support bits) are reserved
   * on all shipped AVX10 parts and are not consulted here. */
  caps->avx10_1_ = false;
  caps->avx10_2_ = false;

  if (avx10_present) {
    const uint32 avx10_version = snapshot->leaf24_0[1] & 0xFFu; ///< EBX[7:0]

    caps->avx10_1_ = (avx10_version >= 1); ///< AVX10.1 or newer
    caps->avx10_2_ = (avx10_version >= 2); ///< AVX10.2 or newer
  }

  /* Composite flags */
  caps->amx_ = ((caps->amx_bf16_ || caps->amx_fp16_ || caps->amx_int8_) != 0);
  caps->vnni_ = ((caps->avx512_vnni_ || caps->avx2_vnni_) != 0);
}

/**
 * @brief Detect CPU SIMD capabilities via the CPUID instruction.
 *
 * @details
 * Thin hardware-reading layer: queries the relevant CPUID leaves through
 * the platform intrinsics, stores the raw registers into a
 * @ref SIMDCpuidSnapshot and delegates all decoding to
 * @ref get_simd_capabilities_from_cpuid().
 *
 * Query order: leaf 1, leaf 7 subleaf 0, leaf 7 subleaf 1, then — only
 * when the AVX10 presence flag (leaf 7 subleaf 1 EDX bit 19) is set —
 * leaf 24H subleaf 0. The guard mirrors the one applied by the decoder:
 * leaf 24H is architecturally undefined on parts without AVX10 and must
 * not be queried there.
 *
 * @param[in,out] caps Pointer to the SIMDCapabilities structure to populate.
 *                  Must be zero-initialised by the caller, since fields
 *                  are only written when their CPUID leaf is available.
 *
 * @pre caps must point to a valid SIMDCapabilities structure.
 * @post All capability flags in caps are set to true/false based
 *       on CPU support. The amx_ and vnni_ composite flags are computed
 *       from their constituent features.
 *
 * @note On non-Windows platforms, if CPUID leaf 1 is not supported,
 *       the function returns early leaving caps untouched.  Callers
 *       should zero-initialise the structure (e.g., @c = @c {})
 *       beforehand.
 *
 * @note This function is platform-specific:
 *       @li Windows (_WIN64): Uses __cpuid() and __cpuidex() from intrin.h
 *       @li Linux/Unix: Uses __get_cpuid() and __cpuid_count() from cpuid.h
 *
 * @see get_simd_capabilities()          Public API accessor.
 * @see get_simd_capabilities_from_cpuid()  Pure decoding layer.
 */
static inline void detect_simd_capabilities(SIMDCapabilities *restrict caps) {

  SIMDCpuidSnapshot snapshot = {};
  uint32 eax, ebx, ecx, edx;

#ifdef _WIN64
  int cpu_info[4];
#endif

  /* CPUID leaf 1 */
#ifdef _WIN64
  __cpuid(cpu_info, 1);
  eax = (uint32)cpu_info[0];
  ebx = (uint32)cpu_info[1];
  ecx = (uint32)cpu_info[2];
  edx = (uint32)cpu_info[3];
#else
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return;
  }
#endif
  snapshot.leaf1[0] = eax;
  snapshot.leaf1[1] = ebx;
  snapshot.leaf1[2] = ecx;
  snapshot.leaf1[3] = edx;

  /* CPUID leaf 7, subleaf 0 */
#ifdef _WIN64
  __cpuidex(cpu_info, 7, 0);
  eax = (uint32)cpu_info[0];
  ebx = (uint32)cpu_info[1];
  ecx = (uint32)cpu_info[2];
  edx = (uint32)cpu_info[3];
#else
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif
  snapshot.leaf7_0[0] = eax;
  snapshot.leaf7_0[1] = ebx;
  snapshot.leaf7_0[2] = ecx;
  snapshot.leaf7_0[3] = edx;

  /* CPUID leaf 7, subleaf 1 */
#ifdef _WIN64
  __cpuidex(cpu_info, 7, 1);
  eax = (uint32)cpu_info[0];
  ebx = (uint32)cpu_info[1];
  ecx = (uint32)cpu_info[2];
  edx = (uint32)cpu_info[3];
#else
  __cpuid_count(7, 1, eax, ebx, ecx, edx);
#endif
  snapshot.leaf7_1[0] = eax;
  snapshot.leaf7_1[1] = ebx;
  snapshot.leaf7_1[2] = ecx;
  snapshot.leaf7_1[3] = edx;

  /* CPUID leaf 24H, subleaf 0: guarded by the AVX10 presence flag
   * (leaf 7 subleaf 1 EDX bit 19), since the leaf is architecturally
   * undefined on parts without AVX10 and must not be queried there. */
  if ((snapshot.leaf7_1[3] & (1 << 19)) != 0) {
#ifdef _WIN64
    __cpuidex(cpu_info, 0x24, 0);
    eax = (uint32)cpu_info[0];
    ebx = (uint32)cpu_info[1];
    ecx = (uint32)cpu_info[2];
    edx = (uint32)cpu_info[3];
#else
    __cpuid_count(0x24, 0, eax, ebx, ecx, edx);
#endif
    snapshot.leaf24_0[0] = eax;
    snapshot.leaf24_0[1] = ebx;
    snapshot.leaf24_0[2] = ecx;
    snapshot.leaf24_0[3] = edx;
  }

  get_simd_capabilities_from_cpuid(&snapshot, caps);
}

#ifdef __linux__
/**
 * @brief One-time initializer for thread-safe lazy initialization.
 *
 * @details
 * This function is registered with call_once() and executes
 * @ref detect_simd_capabilities() to populate the global @ref simd structure.
 * It is guaranteed to be called exactly once, regardless of how many
 * threads invoke @ref get_simd_capabilities() concurrently.
 *
 * @see get_simd_capabilities()
 * @see init_flag
 * @see detect_simd_capabilities()
 */
static inline void init_once(void) { detect_simd_capabilities(&simd); }
#elif defined(_WIN64)
/**
 * @brief Windows one-time initialization callback.
 *
 * @details
 * This function matches the @c PINIT_ONCE_FN signature required by
 * Windows @c InitOnceExecuteOnce().  It delegates to
 * @ref detect_simd_capabilities() to populate the global @ref simd
 * structure.
 *
 * @param[in] once   Pointer to the one-time initialisation structure.
 * @param[in] param  Optional callback data (unused).
 * @param[in] ctx    Optional callback context (unused).
 *
 * @return Always @c TRUE (initialisation always succeeds).
 *
 * @see get_simd_capabilities()
 * @see init_flag
 * @see detect_simd_capabilities()
 */
static BOOL CALLBACK init_once_win(PINIT_ONCE once, PVOID param, PVOID *ctx) {
  (void)once;
  (void)param;
  (void)ctx;
  detect_simd_capabilities(&simd);
  return TRUE;
}
#endif

/**
 * @brief Get CPU capabilities (thread-safe singleton accessor).
 *
 * @details
 * Returns a pointer to the global @ref SIMDCapabilities structure containing
 * all detected SIMD features. The first call triggers detection via
 * @ref detect_simd_capabilities(); subsequent calls return the cached result.
 *
 * @return Pointer to the global SIMDCapabilities structure.
 *         The returned pointer is valid for the lifetime of the process
 *         and must not be freed by the caller.
 *
 * @note This function is thread-safe. The detection is performed at most
 *       once using C11 @c call_once on Linux or @c InitOnceExecuteOnce on
 *       Windows.
 *
 * @par Example:
 * @code{.c}
 *   const SIMDCapabilities *simd = get_simd_capabilities();
 *   if (simd->avx2_) {
 *       // Use AVX2-optimized code path
 *   } else if (simd->sse4_2_) {
 *       // Use SSE4.2 fallback
 *   }
 * @endcode
 *
 * @see SIMDCapabilities
 * @see detect_simd_capabilities()
 * @see init_once()
 */
const SIMDCapabilities *get_simd_capabilities() {
#ifdef __linux__
  call_once(&init_flag, init_once);
#elif defined(_WIN64)
  InitOnceExecuteOnce(&init_flag, init_once_win, nullptr, nullptr);
#endif
  return &simd;
}
