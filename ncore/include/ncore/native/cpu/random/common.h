/**
 * @file common.h
 * @brief Shared plumbing for the CPU random operations.
 */

#pragma once

#include <stdint.h>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/rand/generator.h>
#include <ncore/simd/simd.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

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
 * @brief Operation entry point paired with the instruction set it needs.
 *
 * @remarks
 * @ref base must stay the first member: @ref SIMD_SELECT scans arrays of
 * this type through the embedded descriptor.
 */
typedef struct {
  SimdVariant base; ///< Instruction-set requirement.
  UniformFunc fn;   ///< Operation entry point.
} UniformVariant;

static_assert(alignof(UniformVariant) == alignof(SimdVariant),
              "UniformVariant must match SimdVariant alignment");

#ifdef __cplusplus
}
#endif
