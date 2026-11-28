/**
 * @file simd.h
 * @brief CPU SIMD capability detection and runtime kernel selection interface.
 *
 * @details
 * Runtime detection of CPU SIMD instruction sets and kernel selection
 * keyed on the detected hardware. SIMD capabilities arrive through
 * @ref SIMDCapabilities and @ref get_simd_capabilities().
 *
 * @section simd-support-tiers SIMD Support Tiers
 * The detection follows a hierarchical model:
 *
 * @li SSE4.2 — sse4_2_ — 128-bit — Basic vectorization
 * @li AVX — avx_ — 256-bit — Float ops
 * @li AVX2 — avx2_ — 256-bit — Integer SIMD
 * @li AVX-512 — avx512_* — 512-bit — High-throughput
 * @li AMX — amx_* — Tile — Matrix ops
 *
 * @section thread-safety Thread Safety
 * Detection is performed once on first call.
 * The returned pointer is safe to use from any thread.
 *
 * @note Every flag reflects exactly its own CPUID bit; no cross-feature
 *       gating is applied. VNNI flags (avx2_vnni_, avx512_vnni_) enable
 *       neural network optimizations. The operating-system state required
 *       to execute a vector instruction is checked at dispatch time by
 *       @ref simd_isa_supported(), not while decoding CPUID.
 *
 * @see DetectSIMD.cmake Compile-time SIMD flag detection
 * @see simd.c Implementation of runtime detection and selection
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <ncore/headeronly/macros.h>

/**
 * @struct SIMDCapabilities
 * @brief CPU SIMD capabilities detected at runtime.
 *
 * @details
 * Flags are populated by the first call to @ref get_simd_capabilities()
 * and remain read-only thereafter.
 *
 * @note Composite flags (amx_, vnni_) are OR of their constituent features.
 *
 * @see get_simd_capabilities()
 */
typedef struct {
  bool sse4_2_; ///< Streaming SIMD Extensions 4.2 (128-bit integer/vector ops).
  bool avx_;    ///< Advanced Vector Extensions (256-bit floating-point ops).
  bool avx2_;   ///< AVX2 (256-bit integer SIMD operations).
  bool avx2_int8_; ///< AVX2 VNNI INT8 support (INT8 dot product instructions).
  bool avx2_vnni_; ///< AVX2 VNNI support (Vector Neural Network Instructions).
  bool f16c_;      ///< F16C (FP16 <-> FP32 conversion instructions).

  bool vnni_; ///< Vector Neural Network Instructions (any VNNI support
              ///< available).

  bool fma3_; ///< Fused Multiply-Add 3 (FMA3 instructions).

  bool avx512f_;     ///< AVX-512 Foundation (512-bit vector operations).
  bool avx512_bw_;   ///< AVX-512 Byte/Word (8-bit and 16-bit integer ops).
  bool avx512_dq_;   ///< AVX-512 Doubleword/Quadword (32-bit and 64-bit integer
                     ///< ops).
  bool avx512_vl_;   ///< AVX-512 Vector Length extensions (128/256-bit
                     ///< compatibility).
  bool avx512_vnni_; ///< AVX-512 VNNI (Vector Neural Network Instructions).
  bool avx512_fp16_; ///< AVX-512 FP16 (half-precision floating-point ops).
  bool avx512_bf16_; ///< AVX-512 BF16 (bfloat16 dot product instructions).

  bool avx10_1_; ///< AVX-10.1 (converged AVX-512 baseline; version >= 1).
  bool avx10_2_; ///< AVX-10.2 (mandatory 512-bit width; version >= 2).

  bool amx_;      ///< Any AMX tile support (composite flag).
  bool amx_fp16_; ///< AMX FP16 tile support (half-precision matrix ops).
  bool amx_bf16_; ///< AMX BF16 tile support (bfloat16 matrix ops).
  bool amx_int8_; ///< AMX INT8 tile support (int8 matrix multiply).

  bool os_xsave_; ///< OSXSAVE (CPUID.1:ECX[27]): XGETBV is legal and the
                  ///< operating system enabled the XSAVE feature set.
  uint64_t xcr0_; ///< XCR0 state mask read through XGETBV(0). Zero when
                  ///< @ref os_xsave_ is false. Not a CPUID feature flag.
} SIMDCapabilities;

/**
 * @struct SIMDCpuidSnapshot
 * @brief Raw CPUID register values consumed by
 *        @ref get_simd_capabilities_from_cpuid().
 *
 * @details
 * Each member holds the four output registers of one CPUID query in
 * hardware order: EAX, EBX, ECX, EDX.
 *
 * The snapshot decouples the pure bit-to-flag mapping (unit-testable with
 * synthetic snapshots of processors that are not physically available)
 * from the platform-specific register reads performed by
 * @ref get_simd_capabilities().
 *
 * @see get_simd_capabilities_from_cpuid()
 */
typedef struct {
  uint32_t leaf1[4];    ///< Output registers of CPUID.(EAX=01H).
  uint32_t leaf7_0[4];  ///< Output registers of CPUID.(EAX=07H,ECX=00H).
  uint32_t leaf7_1[4];  ///< Output registers of CPUID.(EAX=07H,ECX=01H).
  uint32_t leaf24_0[4]; ///< Output registers of CPUID.(EAX=24H,ECX=00H).
  uint64_t xcr0;        ///< XCR0 read through XGETBV(0); zero when
                        ///< OSXSAVE is clear. Not a CPUID output.
} SIMDCpuidSnapshot;

/**
 * @typedef SimdIsa
 * @brief Bit mask naming the instruction sets a kernel variant requires.
 *
 * @details
 * Every bit mirrors one feature flag of @ref SIMDCapabilities, so a
 * requirement masks off exactly the hardware facts the kernel needs.
 * The mask semantic is conjunction: a variant runs only when every bit
 * it names is present. Composite aliases are ORs of real bits, never a
 * new bit.
 *
 * FMA and F16C have their own bits and are never inferred from AVX or
 * AVX2: a CPU can enumerate AVX2 without the FMA3 or F16C state the
 * kernel was compiled against.
 *
 * @see simd_isa_supported()
 * @see simd_select()
 */
typedef uint32_t SimdIsa;

/** @brief Instruction-set bits used by @ref SimdIsa. */
enum : uint32_t {
  SimdIsaNone = 0u,             ///< No requirement; always selectable.
  SimdIsaSSE4_2 = 1u << 0,      ///< SSE4.2.
  SimdIsaAVX = 1u << 1,         ///< AVX.
  SimdIsaAVX2 = 1u << 2,        ///< AVX2.
  SimdIsaAVX2Int8 = 1u << 3,    ///< AVX2 VNNI INT8.
  SimdIsaAVX2VNNI = 1u << 4,    ///< AVX2 VNNI.
  SimdIsaF16C = 1u << 5,        ///< F16C.
  SimdIsaVNNI = 1u << 6,        ///< Any VNNI flavor (composite flag). Variant
                                ///< masks should name AVX2VNNI or AVX512VNNI
                                ///< instead: the required OS state differs.
  SimdIsaFMA3 = 1u << 7,        ///< FMA3.
  SimdIsaAVX512F = 1u << 8,     ///< AVX-512 Foundation.
  SimdIsaAVX512BW = 1u << 9,    ///< AVX-512 Byte/Word.
  SimdIsaAVX512DQ = 1u << 10,   ///< AVX-512 Doubleword/Quadword.
  SimdIsaAVX512VL = 1u << 11,   ///< AVX-512 Vector Length extensions.
  SimdIsaAVX512VNNI = 1u << 12, ///< AVX-512 VNNI.
  SimdIsaAVX512FP16 = 1u << 13, ///< AVX-512 FP16.
  SimdIsaAVX512BF16 = 1u << 14, ///< AVX-512 BF16.
  SimdIsaAVX10_1 = 1u << 15,    ///< AVX10.1 or newer.
  SimdIsaAVX10_2 = 1u << 16,    ///< AVX10.2 or newer.
  SimdIsaAMX = 1u << 17,        ///< Any AMX tile support (composite flag).
  SimdIsaAMXFP16 = 1u << 18,    ///< AMX FP16.
  SimdIsaAMXBF16 = 1u << 19,    ///< AMX BF16.
  SimdIsaAMXINT8 = 1u << 20,    ///< AMX INT8.

  SimdIsaAVX2FMA = SimdIsaAVX | SimdIsaAVX2 | SimdIsaFMA3,
  SimdIsaAVX2FMAF16C = SimdIsaAVX2FMA | SimdIsaF16C,
  SimdIsaAVX512FDQ = SimdIsaAVX512F | SimdIsaAVX512DQ,
  SimdIsaAVX512FBF16 = SimdIsaAVX512F | SimdIsaAVX512BF16,

  SimdIsaAll = (1u << 21) - 1u, ///< Union of every instruction-set bit.
};

/**
 * @struct SimdVariant
 * @brief Generic dispatch descriptor: the instruction set one variant needs.
 *
 * @details
 * Typed tables embed this structure as their first member, so the address
 * of a typed entry and the address of its descriptor coincide. The alignment
 * matches those entries, whose next member is a function pointer, so the
 * descriptor pointer can be reinterpreted as a typed entry. Iterate such
 * tables through @ref SIMD_SELECT, which carries the entry stride. A table
 * whose last entry requires @ref SimdIsaNone always yields a fallback.
 *
 * @see SIMD_SELECT
 * @see simd_select()
 */
typedef struct {
  ALIGN(8) SimdIsa isa; ///< Instruction sets the variant requires.
} SimdVariant;

static_assert(alignof(SimdVariant) >= alignof(void *),
              "SimdVariant must match the alignment of typed variant entries");

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get CPU capabilities (thread-safe singleton accessor).
 *
 * @details
 * Detection runs once; subsequent calls return the cached result. The
 * first call uses C11 @c call_once on Linux or @c InitOnceExecuteOnce
 * on Windows.
 *
 * @return Pointer to the global SIMDCapabilities structure.
 *         The returned pointer is valid for the lifetime of the process
 *         and must not be freed by the caller.
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
 */
const SIMDCapabilities *get_simd_capabilities();

/**
 * @brief Maps raw CPUID registers to SIMD capabilities.
 *
 * @details
 * Pure, deterministic translation of a @ref SIMDCpuidSnapshot into a
 * @ref SIMDCapabilities structure. It performs no hardware access, which
 * makes it directly unit-testable against synthetic snapshots of
 * processors that are not physically available to the build host.
 *
 * The mapping follows the Intel SDM layout:
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
 * @li Leaf 1 ECX bit 27 feeds os_xsave_ and @c snapshot->xcr0 feeds
 *     xcr0_ verbatim.
 *
 * No cross-feature gating is applied: every flag is set exactly when its
 * source bit is set (e.g. avx512_fp16_ does not require avx512f_).
 *
 * @param[in]     snapshot  Raw CPUID registers and XCR0. Must not be null.
 * @param[in,out] caps      Destination capabilities. Must not be null;
 *                          every field is overwritten.
 *
 * @see get_simd_capabilities()
 */
void get_simd_capabilities_from_cpuid(const SIMDCpuidSnapshot *snapshot,
                                      SIMDCapabilities *caps);

/**
 * @brief Report whether the host can execute every instruction set in @p isa.
 *
 * @details
 * Checks the feature bits of @p caps and, for vector code, the
 * operating-system state that makes those instructions legal:
 *
 * @li AVX, AVX2, FMA3, F16C and AVX2 VNNI need XCR0[2:1] = 11b, which the
 *     OS sets only after it has enabled XSAVE for the AVX state (Intel SDM
 *     Vol. 1, 13.3). Executing them without that state raises #UD, so the
 *     CPUID feature bit alone is not enough.
 * @li AVX-512 and AVX10 need XCR0[7:5] = 111b for the opmask and ZMM state
 *     (Intel SDM Vol. 1, 15.2; Intel AVX10.2 Architecture Specification,
 *     3.1.3).
 * @li AMX needs XCR0[18:17] = 11b for the tile configuration and tile data
 *     state (Linux x86 FPU state definitions, arch/x86/include/asm/fpu/
 *     types.h: XFEATURE_XTILE_CFG = 17, XFEATURE_XTILE_DATA = 18; see also
 *     Documentation/arch/x86/xstate.rst). The AVX state check applies to
 *     AMX masks as a conservative precondition.
 *
 * The XCR0 checks also require os_xsave_, since XGETBV is illegal when
 * OSXSAVE is clear. On Linux, AMX additionally requires the process to
 * request tile-data permission through @c arch_prctl; the state bits only
 * prove the kernel reserved the XSAVE area.
 *
 * @param[in] isa  Instruction-set mask to test. @ref SimdIsaNone always
 *                 succeeds; bits outside @ref SimdIsaAll always fail.
 * @param[in] caps Capabilities to test against. Must not be null.
 *
 * @return @c true when every named instruction set is usable.
 */
bool simd_isa_supported(SimdIsa isa, const SIMDCapabilities *caps);

/**
 * @brief Select the first variant whose requirement @p caps satisfies.
 *
 * @details
 * Strided form used by typed tables. Entries are scanned from index 0,
 * which callers order from fastest to most portable. A typed table
 * embeds @ref SimdVariant as its first member and ends with a
 * @ref SimdIsaNone entry, so the pointer returned for a non-empty table
 * always names a callable variant.
 *
 * @param[in] variants First entry of the table. Must not be null when
 *                     @p count is greater than zero.
 * @param[in] stride   Size in bytes of one typed entry. Must be a multiple
 *                     of @c alignof(SimdVariant).
 * @param[in] count    Number of entries in the table.
 * @param[in] caps     Capabilities to test against. Must not be null.
 *
 * @return Address of the selected entry's @ref SimdVariant, or null when
 *         no entry is supported.
 *
 * @see SIMD_SELECT
 */
const SimdVariant *simd_select_stride(const SimdVariant variants[],
                                      size_t stride, size_t count,
                                      const SIMDCapabilities *caps);

/**
 * @brief Select the first supported entry of a @ref SimdVariant table.
 *
 * @details
 * Unstrided convenience form for tables whose entries are exactly
 * @ref SimdVariant. Typed tables use @ref SIMD_SELECT instead.
 *
 * @param[in] variants Table entries. Must not be null when @p count is
 *                     greater than zero.
 * @param[in] count    Number of entries.
 * @param[in] caps     Capabilities to test against. Must not be null.
 *
 * @return The selected entry, or null when none is supported.
 */
const SimdVariant *simd_select(const SimdVariant variants[], size_t count,
                               const SIMDCapabilities *caps);

/**
 * @brief Render @p isa as a readable, thread-local string.
 *
 * @details
 * Produces names such as @c "avx | avx2 | f16c". Returns @c "none" for
 * @ref SimdIsaNone and @c "unknown" when the mask carries bits outside
 * @ref SimdIsaAll. The returned pointer belongs to a per-thread buffer
 * and is overwritten by the next call on the same thread.
 *
 * @param[in] isa  Instruction-set mask to render.
 *
 * @return Constant string for @c none and @c unknown, or the thread-local
 *         buffer for composed masks.
 */
const char *simd_isa_name(SimdIsa isa);
#ifdef __cplusplus
}
#endif

/**
 * @def SIMD_VARIANT_COUNT
 * @brief Number of entries in a statically sized variant table.
 */
#define SIMD_VARIANT_COUNT(variants) (sizeof(variants) / sizeof((variants)[0]))

/**
 * @def SIMD_SELECT
 * @brief Select from a typed table whose first member is @ref SimdVariant.
 *
 * @details
 * Expands to @ref simd_select_stride() with the stride and address of the
 * first descriptor taken from @p variants. The result points at the
 * descriptor inside the selected typed entry; a reinterpreting cast
 * recovers the typed pointer.
 *
 * @param variants Typed table (array or pointer to its first entry).
 * @param count    Number of entries.
 * @param caps     Capabilities to test against. Must not be null.
 */
#define SIMD_SELECT(variants, count, caps)                                     \
  simd_select_stride(&(variants)[0].base, sizeof((variants)[0]), (count),      \
                     (caps))
