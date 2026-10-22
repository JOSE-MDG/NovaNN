/**
 * @file common.h
 * @brief Shared plumbing for the CPU random operations.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/rand/generator.h>
#include <ncore/simd/simd.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum UniformISA_
 * @brief Instruction set a CPU random operation variant requires.
 *
 * @details
 * Each value names one capability set the host must provide before the
 * matching variant may run. The sets mirror the ones the dtype-cast
 * operations dispatch on, so both families read the same hardware facts.
 */
typedef enum UniformISA_ : uint8_t {
  UniformISANone = 0,    ///< Portable fallback, no instruction set needed.
  UniformISAAVX2FMA,     ///< AVX/AVX2 with FMA3 (256-bit vectors).
  UniformISAAVX2FMAF16C, ///< AVX/AVX2 with FMA3 and F16C.
  UniformISAAVX512F,     ///< AVX-512 Foundation (512-bit vectors).
  UniformISAAVX512DQ,    ///< AVX-512 Foundation with Doubleword/Quadword.
  UniformISAAVX512BF16,  ///< AVX-512 Foundation with BF16.
} UniformISA_;

/**
 * @typedef UniformFunc
 * @brief Signature of a CPU uniform operation.
 *
 * @details
 * The operation fills the destination in place and reports through the
 * status pointer.
 *
 * @param[in]  ten        Destination tensor, filled in place. Must be a
 *                        live CPU tensor of a floating point type whose
 *                        elements are contiguous.
 * @param[in]  low        Lower bound of the sampling interval.
 * @param[in]  high       Exclusive upper bound of the sampling interval.
 * @param[in]  generator  Private generator, or null for the process
 *                        default. Must live on the tensor's device.
 * @param[out] st         Caller-owned status. Must not be @c nullptr.
 */
typedef void (*UniformFunc)(Tensor *, float, float, Generator_ *,
                            novaStatus_t *);

/**
 * @struct UniformVariant
 * @brief Variant paired with the instruction set it needs.
 */
typedef struct {
  UniformFunc fn;  ///< Operation entry point.
  UniformISA_ isa; ///< Instruction set the operation requires.
} UniformVariant;

/**
 * @brief Report whether the host CPU provides @p isa.
 *
 * @param[in] isa  Capability set the caller needs.
 *
 * @return @c true when every capability in @p isa is present.
 */
static inline bool uniform_isa_supported(UniformISA_ isa) {
  const SIMDCapabilities *caps = get_simd_capabilities();
  switch (isa) {
  case UniformISAAVX2FMA:
    return caps->avx_ && caps->avx2_ && caps->fma3_;
  case UniformISAAVX2FMAF16C:
    return caps->avx_ && caps->avx2_ && caps->fma3_ && caps->f16c_;
  case UniformISAAVX512F:
    return caps->avx512f_;
  case UniformISAAVX512DQ:
    return caps->avx512f_ && caps->avx512_dq_;
  case UniformISAAVX512BF16:
    return caps->avx512f_ && caps->avx512_bf16_;
  case UniformISANone:
  default:
    break;
  }
  return true;
}

/**
 * @brief Draw @p n raw 32-bit words for @p device.
 *
 * @details
 * A private generator advances in place. Without one the live process
 * default is consumed, so successive fills hand out fresh values
 * instead of replaying the frozen snapshot a copy would carry.
 *
 * @param[in]  device     Placement the words are drawn for.
 * @param[in]  generator  Private generator, or null for the default.
 * @param[out] out        Destination words.
 * @param[in]  n          Words to draw.
 *
 * @return Status with @c err set to @c novaSuccess on success.
 */
static inline novaStatus_t uniform_fill_u32(Device_ device,
                                            Generator_ *generator,
                                            uint32_t *out, size_t n) {
  if (generator != nullptr) {
    return generator_fill_u32(generator, out, n);
  }
  return default_fill_u32(device, out, n);
}

/**
 * @brief Draw @p n raw 64-bit words for @p device.
 *
 * @copydetails uniform_fill_u32()
 *
 * @param[in]  device     Placement the words are drawn for.
 * @param[in]  generator  Private generator, or null for the default.
 * @param[out] out        Destination words.
 * @param[in]  n          Words to draw.
 *
 * @return Status with @c err set to @c novaSuccess on success.
 */
static inline novaStatus_t uniform_fill_u64(Device_ device,
                                            Generator_ *generator,
                                            uint64_t *out, size_t n) {
  if (generator != nullptr) {
    return generator_fill_u64(generator, out, n);
  }
  return default_fill_u64(device, out, n);
}

#ifdef __cplusplus
}
#endif
