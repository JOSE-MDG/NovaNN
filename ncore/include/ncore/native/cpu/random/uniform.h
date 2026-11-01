/**
 * @file uniform.h
 * @brief CPU uniform operation entry point and per-ISA variants.
 *
 * @details
 * The operation fills a tensor that already exists: the storage is
 * neither replaced nor resized. The per-ISA variants are declared
 * here so a harness can exercise each instruction set on hosts that
 * do not run it.
 *
 * @see common.h
 */

#pragma once

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/native/cpu/random/common.h>
#include <ncore/rand/generator.h>
#include <ncore/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fill @p ten with uniform samples drawn from [low, high).
 *
 * @details
 * The samples land in the tensor's own storage, so the call is
 * in-place. Without a private generator the live process default is
 * consumed, which advances the shared stream.
 *
 * The caller provides a live CPU tensor of a floating point type
 * and a generator that lives on the same device.
 *
 * @param[in,out] ten       Destination tensor, filled in place.
 * @param[in]     low       Lower bound of the sampling interval.
 * @param[in]     high      Exclusive upper bound of the sampling interval.
 * @param[in]     generator Private generator, or null for the process
 *                          default.
 *
 * @return Status with @c err set to @c novaSuccess on success.
 */
novaStatus_t uniform_cpu_impl(Tensor *ten, float low, float high,
                              Generator_ *generator);

/**
 * @brief Fill @p ten with uniform float32 samples using AVX-512.
 */
void uniform_cpu_impl_f32_avx512(Tensor *ten, float low, float high,
                                 Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform float32 samples using AVX/AVX2.
 */
void uniform_cpu_impl_f32_avx_avx2_fma(Tensor *ten, float low, float high,
                                       Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform float32 samples using portable code.
 */
void uniform_cpu_impl_f32_scalar(Tensor *ten, float low, float high,
                                 Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform float64 samples using AVX-512.
 */
void uniform_cpu_impl_f64_avx512dq(Tensor *ten, float low, float high,
                                   Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform float64 samples using AVX/AVX2.
 */
void uniform_cpu_impl_f64_avx_avx2_fma(Tensor *ten, float low, float high,
                                       Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform float64 samples using portable code.
 */
void uniform_cpu_impl_f64_scalar(Tensor *ten, float low, float high,
                                 Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform float16 samples using AVX-512.
 */
void uniform_cpu_impl_fp16_avx512(Tensor *ten, float low, float high,
                                  Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform float16 samples using AVX/AVX2 and F16C.
 */
void uniform_cpu_impl_fp16_avx_avx2_fma_f16c(Tensor *ten, float low, float high,
                                             Generator_ *generator,
                                             novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform float16 samples using portable code.
 */
void uniform_cpu_impl_fp16_scalar(Tensor *ten, float low, float high,
                                  Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform bfloat16 samples using AVX-512.
 */
void uniform_cpu_impl_bf16_avx512bf16(Tensor *ten, float low, float high,
                                      Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform bfloat16 samples using portable code.
 */
void uniform_cpu_impl_bf16_scalar(Tensor *ten, float low, float high,
                                  Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform float8_e4m3fn samples using portable code.
 */
void uniform_cpu_impl_fp8e4m3_scalar(Tensor *ten, float low, float high,
                                     Generator_ *generator, novaStatus_t *st);

/**
 * @brief Fill @p ten with uniform float8_e5m2 samples using portable code.
 */
void uniform_cpu_impl_fp8e5m2_scalar(Tensor *ten, float low, float high,
                                     Generator_ *generator, novaStatus_t *st);

#ifdef __cplusplus
}
#endif
