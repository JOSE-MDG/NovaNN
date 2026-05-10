#pragma once

#include <stdbool.h>

/**
 * @brief CPU SIMD capabilities detected at runtime.
 *
 * Contains flags indicating which SIMD instruction sets are available
 * on the current processor. These are used to select optimized kernels.
 */
typedef struct {
  bool sse4_2_;    ///< Streaming SIMD Extensions 4.2
  bool avx_;       ///< Advanced Vector Extensions
  bool avx2_;      ///< AVX2 (256-bit integer SIMD)
  bool avx2_int8_; ///< AVX2 VNNI INT8 support
  bool avx2_vnni_; ///< AVX2 VNNI support
  bool f16c_;      ///< F16C (FP16 conversion)

  bool vnni_;      ///< Vector Neural Network Instructions (any)

  bool fma3_;      ///< Fused Multiply-Add 3

  bool avx512f_;       ///< AVX-512 Foundation
  bool avx512_bw_;     ///< AVX-512 Byte/Word
  bool avx512_dq_;     ///< AVX-512 Doubleword/Quadword
  bool avx512_vl_;     ///< AVX-512 Vector Length
  bool avx512_vnni_;   ///< AVX-512 VNNI
  bool avx512_fp16_;   ///< AVX-512 FP16
  bool avx512_bf16_;   ///< AVX-512 BF16

  bool amx_;       ///< Any AMX tile support
  bool amx_fp16_;  ///< AMX FP16 tile support
  bool amx_bf16_;  ///< AMX BF16 tile support
  bool amx_int8_;  ///< AMX INT8 tile support
} Capabilities_;

/**
 * @brief Get CPU SIMD capabilities (thread-safe, singleton).
 * @return Pointer to global Capabilities_ structure.
 *
 * On first call, detects and caches CPU capabilities. Subsequent calls
 * return the cached result. Thread-safe via call_once.
 */
const Capabilities_ *get_cpu_capabilities();
