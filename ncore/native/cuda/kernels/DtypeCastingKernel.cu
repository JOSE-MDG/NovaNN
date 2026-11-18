/**
 * @file DtypeCastingKernel.cu
 * @brief CUDA kernels for tensor dtype casting.
 *
 * @details
 * Covers all 212 supported dtype conversions with three kernel shapes
 * (plain one to one, FP4 unpack, FP4 pack) instantiated per pair and
 * selected at run time through a dtype keyed map in
 * @ref launchCudaDtypeCastingKernel. Every kernel is a grid stride loop
 * that moves whole 16B transactions when both pointers prove a
 * 16B-aligned run (plain: W = 16 / narrower item elements per chunk;
 * unpack: eight bytes into sixteen elements; pack: sixteen elements
 * into 8/S bytes), with a masked scalar loop covering the tail and the
 * misaligned case, so any grid size and any pointer stay correct.
 * Low precision float paths use hardware intrinsics for FP16
 * and BF16; FP8 and FP4 reuse the shared soft float helpers from
 * @ref ncore::dtypes::detail, so device conversions follow the same
 * rules as the host scalar kernels (saturating narrowing, NaN to zero
 * for float to integer, round to nearest even packing). The FP8 E8M0
 * scale-factor pair keeps the host semantics exactly: exponent
 * round-to-nearest-even on narrowing and the subnormal 2^-127 widening
 * on expansion.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/dtypes/fp4_e2m1fn_x2.hh>
#include <ncore/headeronly/dtypes/fp8_e4m3fn.hh>
#include <ncore/headeronly/dtypes/fp8_e5m2.hh>
#include <ncore/headeronly/dtypes/fp8_e8m0fnu.hh>
#include <ncore/headeronly/heuristics/kernels/launch_config.hh>
#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>

namespace hk = ncore::heuristics::kernels;

#include "DtypeCastingKernel.h"
#include "memory/CudaEngineStream.hpp"

#ifndef __CUDA_ARCH__
#include <unordered_map>

#include "DetectCudaDeviceInfo.hpp"
#endif

#define OK                                                                     \
  {.err = novaSuccess, .message = nova_get_error_msg(novaSuccess, nullptr)}

namespace {

using F16T = __half;
using BF16T = __nv_bfloat16;

/// Half bit test and x86 F16C-equivalent NaN widening.
__device__ inline bool f16IsNan(uint16_t h) {
  return ((h >> 10U) & 0x1FU) == 0x1FU && (h & 0x3FFU) != 0U;
}

__device__ inline float f16NanToF32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) >> 15U) << 31U;
  const uint32_t payload = static_cast<uint32_t>(h) & 0x3FFU;
  uint32_t mant = payload << 13U;
  if ((payload & 0x200U) == 0U) {
    mant |= 0x400000U;
  }
  return __uint_as_float(sign | 0x7F800000U | mant);
}

__device__ inline float f16BitsToF32(uint16_t h) {
  if (f16IsNan(h)) {
    return f16NanToF32(h);
  }
  return __half2float(__ushort_as_half(h));
}

/// @brief Conversion shape of one dtype pair.
enum class CastKind : uint8_t { kPlain, kUnpack, kPack };

__host__ __device__ constexpr bool isIntDType(DType_ d) {
  return d >= DType_::Signed8;
}

__host__ __device__ constexpr bool isSignedInt(DType_ d) {
  return d == DType_::Signed8 || d == DType_::Signed16 ||
         d == DType_::Signed32 || d == DType_::Signed64;
}

// Reduced precision conversions reuse the shared soft float helpers.
namespace dd = ncore::dtypes::detail;

// Source element to float32.
template <DType_ S, typename ST> struct ToF32;
template <> struct ToF32<DType_::Float32, float> {
  static __device__ float convert(float v) { return v; }
};
template <> struct ToF32<DType_::Float64, double> {
  static __device__ float convert(double v) { return static_cast<float>(v); }
};
template <> struct ToF32<DType_::Float16, F16T> {
  static __device__ float convert(F16T v) {
    return f16BitsToF32(__half_as_ushort(v));
  }
};
template <> struct ToF32<DType_::BFloat16, BF16T> {
  static __device__ float convert(BF16T v) {
    uint16_t bits = __bfloat16_as_ushort(v);
    if (((bits >> 7U) & 0xFFU) == 0xFFU && (bits & 0x7FU) != 0U &&
        (bits & 0x40U) == 0U) {
      // Replicate x86 widening: the intrinsic preserves signaling NaN
      // patterns, while the CPU reference quietizes them.
      bits |= 0x40U;
      const BF16T quiet = __ushort_as_bfloat16(bits);
      return __bfloat162float(quiet);
    }
    return __bfloat162float(v);
  }
};
template <> struct ToF32<DType_::Float8E4M3fn, uint8_t> {
  static __device__ float convert(uint8_t v) {
    return dd::fp8e4m3fn_to_fp32_value(v);
  }
};
template <> struct ToF32<DType_::Float8E5M2, uint8_t> {
  static __device__ float convert(uint8_t v) {
    // Widen into the half domain and reuse the F16C-equivalent path:
    // E5M2 shares the FP16 exponent layout exactly.
    const uint16_t wide = static_cast<uint16_t>(static_cast<uint16_t>(v) << 8U);
    return f16BitsToF32(wide);
  }
};
template <> struct ToF32<DType_::Float8E8M0fnu, uint8_t> {
  static __device__ float convert(uint8_t v) {
    return dd::fp8e8m0fnu_to_fp32_value(v);
  }
};
template <> struct ToF32<DType_::Signed8, int8_t> {
  static __device__ float convert(int8_t v) { return static_cast<float>(v); }
};
template <> struct ToF32<DType_::UnSigned8, uint8_t> {
  static __device__ float convert(uint8_t v) { return static_cast<float>(v); }
};
template <> struct ToF32<DType_::Signed16, int16_t> {
  static __device__ float convert(int16_t v) { return static_cast<float>(v); }
};
template <> struct ToF32<DType_::UnSigned16, uint16_t> {
  static __device__ float convert(uint16_t v) { return static_cast<float>(v); }
};
template <> struct ToF32<DType_::Signed32, int32_t> {
  static __device__ float convert(int32_t v) { return static_cast<float>(v); }
};
template <> struct ToF32<DType_::UnSigned32, uint32_t> {
  static __device__ float convert(uint32_t v) { return static_cast<float>(v); }
};
template <> struct ToF32<DType_::Signed64, int64_t> {
  static __device__ float convert(int64_t v) { return static_cast<float>(v); }
};
template <> struct ToF32<DType_::UnSigned64, uint64_t> {
  static __device__ float convert(uint64_t v) { return static_cast<float>(v); }
};

/// @brief Source element to float64
template <DType_ S, typename ST> struct ToF64;
template <> struct ToF64<DType_::Float32, float> {
  static __device__ double convert(float v) { return static_cast<double>(v); }
};
template <> struct ToF64<DType_::Float64, double> {
  static __device__ double convert(double v) { return v; }
};
template <> struct ToF64<DType_::Float16, F16T> {
  static __device__ double convert(F16T v) {
    return static_cast<double>(ToF32<DType_::Float16, F16T>::convert(v));
  }
};
template <> struct ToF64<DType_::BFloat16, BF16T> {
  static __device__ double convert(BF16T v) {
    return static_cast<double>(ToF32<DType_::BFloat16, BF16T>::convert(v));
  }
};
template <> struct ToF64<DType_::Float8E4M3fn, uint8_t> {
  static __device__ double convert(uint8_t v) {
    return static_cast<double>(dd::fp8e4m3fn_to_fp32_value(v));
  }
};
template <> struct ToF64<DType_::Float8E5M2, uint8_t> {
  static __device__ double convert(uint8_t v) {
    return static_cast<double>(ToF32<DType_::Float8E5M2, uint8_t>::convert(v));
  }
};
template <> struct ToF64<DType_::Signed8, int8_t> {
  static __device__ double convert(int8_t v) { return static_cast<double>(v); }
};
template <> struct ToF64<DType_::UnSigned8, uint8_t> {
  static __device__ double convert(uint8_t v) { return static_cast<double>(v); }
};
template <> struct ToF64<DType_::Signed16, int16_t> {
  static __device__ double convert(int16_t v) { return static_cast<double>(v); }
};
template <> struct ToF64<DType_::UnSigned16, uint16_t> {
  static __device__ double convert(uint16_t v) {
    return static_cast<double>(v);
  }
};
template <> struct ToF64<DType_::Signed32, int32_t> {
  static __device__ double convert(int32_t v) { return static_cast<double>(v); }
};
template <> struct ToF64<DType_::UnSigned32, uint32_t> {
  static __device__ double convert(uint32_t v) {
    return static_cast<double>(v);
  }
};
template <> struct ToF64<DType_::Signed64, int64_t> {
  static __device__ double convert(int64_t v) { return static_cast<double>(v); }
};
template <> struct ToF64<DType_::UnSigned64, uint64_t> {
  static __device__ double convert(uint64_t v) {
    return static_cast<double>(v);
  }
};

/**
 * @brief Float32 to destination element.
 *
 * NaN maps to zero for integer destinations.
 */

template <DType_ D, typename DT> struct FromF32;
template <> struct FromF32<DType_::Float32, float> {
  static __device__ float convert(float v) { return v; }
};
template <> struct FromF32<DType_::Float64, double> {
  static __device__ double convert(float v) { return static_cast<double>(v); }
};
template <> struct FromF32<DType_::Float16, F16T> {
  static __device__ F16T convert(float v) {
    const uint32_t w = dd::fp32_to_bits(v);
    if (((w >> 23U) & 0xFFU) == 0xFFU && (w & 0x7FFFFFU) != 0U) {
      // Replicate x86 narrowing: __float2half drops the sign of NaN
      // inputs, while the CPU reference preserves it and shifts the
      // payload, quietizing signaling inputs.
      const uint32_t sign = (w >> 31U) << 15U;
      const uint32_t payload = w & 0x7FFFFFU;
      uint32_t mant = payload >> 13U;
      if ((payload & 0x400000U) == 0U) {
        mant |= 0x200U;
      }
      return __ushort_as_half(static_cast<uint16_t>(sign | 0x7C00U | mant));
    }
    return __float2half(v);
  }
};
template <> struct FromF32<DType_::BFloat16, BF16T> {
  static __device__ BF16T convert(float v) {
    if (isnan(v)) {
      return __ushort_as_bfloat16(UINT16_C(0x7FC0));
    }
    return __float2bfloat16_rn(v);
  }
};
template <> struct FromF32<DType_::Float8E4M3fn, uint8_t> {
  static __device__ uint8_t convert(float v) {
    return dd::fp8e4m3fn_from_fp32_value(v);
  }
};
template <> struct FromF32<DType_::Float8E5M2, uint8_t> {
  static __device__ uint8_t convert(float v) {
    return dd::fp8e5m2_from_fp32_value(v);
  }
};
template <> struct FromF32<DType_::Float8E8M0fnu, uint8_t> {
  static __device__ uint8_t convert(float v) {
    return dd::fp8e8m0fnu_from_fp32_value(v);
  }
};
template <> struct FromF32<DType_::Signed8, int8_t> {
  static __device__ int8_t convert(float v) {
    if (isnan(v)) {
      return static_cast<int8_t>(0);
    }
    const float c = v < -128.0F ? -128.0F : (v > 127.0F ? 127.0F : v);
    return static_cast<int8_t>(c);
  }
};
template <> struct FromF32<DType_::UnSigned8, uint8_t> {
  static __device__ uint8_t convert(float v) {
    if (isnan(v)) {
      return static_cast<uint8_t>(0);
    }
    const float c = v <= 0.0F ? 0.0F : (v > 255.0F ? 255.0F : v);
    return static_cast<uint8_t>(c);
  }
};
template <> struct FromF32<DType_::Signed16, int16_t> {
  static __device__ int16_t convert(float v) {
    if (isnan(v)) {
      return static_cast<int16_t>(0);
    }
    const float c = v < -32768.0F ? -32768.0F : (v > 32767.0F ? 32767.0F : v);
    return static_cast<int16_t>(c);
  }
};
template <> struct FromF32<DType_::UnSigned16, uint16_t> {
  static __device__ uint16_t convert(float v) {
    if (isnan(v)) {
      return static_cast<uint16_t>(0);
    }
    const float c = v <= 0.0F ? 0.0F : (v > 65535.0F ? 65535.0F : v);
    return static_cast<uint16_t>(c);
  }
};
template <> struct FromF32<DType_::Signed32, int32_t> {
  static __device__ int32_t convert(float v) {
    if (isnan(v)) {
      return static_cast<int32_t>(0);
    }
    const float c = v < -2147483648.0F
                        ? -2147483648.0F
                        : (v > 2147483520.0F ? 2147483520.0F : v);
    return static_cast<int32_t>(c);
  }
};
template <> struct FromF32<DType_::UnSigned32, uint32_t> {
  static __device__ uint32_t convert(float v) {
    if (isnan(v)) {
      return static_cast<uint32_t>(0);
    }
    const float c = v <= 0.0F ? 0.0F : (v > 4294967040.0F ? 4294967040.0F : v);
    return static_cast<uint32_t>(c);
  }
};
template <> struct FromF32<DType_::Signed64, int64_t> {
  static __device__ int64_t convert(float v) {
    if (isnan(v)) {
      return static_cast<int64_t>(0);
    }
    const float c =
        v < -9223372036854775808.0F
            ? -9223372036854775808.0F
            : (v > 9223371487098961920.0F ? 9223371487098961920.0F : v);
    return static_cast<int64_t>(c);
  }
};
template <> struct FromF32<DType_::UnSigned64, uint64_t> {
  static __device__ uint64_t convert(float v) {
    if (isnan(v)) {
      return static_cast<uint64_t>(0);
    }
    const float c =
        v <= 0.0F ? 0.0F
                  : (v > 18446742974197923840.0F ? 18446742974197923840.0F : v);
    return static_cast<uint64_t>(c);
  }
};

/**
 * @brief Float64 to destination element.
 *
 * Integer destinations clamp at double precision bounds.
 */
template <DType_ D, typename DT> struct FromF64;
template <> struct FromF64<DType_::Float32, float> {
  static __device__ float convert(double v) { return static_cast<float>(v); }
};
template <> struct FromF64<DType_::Float64, double> {
  static __device__ double convert(double v) { return v; }
};
template <> struct FromF64<DType_::Float16, F16T> {
  static __device__ F16T convert(double v) {
    return FromF32<DType_::Float16, F16T>::convert(static_cast<float>(v));
  }
};
template <> struct FromF64<DType_::BFloat16, BF16T> {
  static __device__ BF16T convert(double v) {
    return FromF32<DType_::BFloat16, BF16T>::convert(static_cast<float>(v));
  }
};
template <> struct FromF64<DType_::Float8E4M3fn, uint8_t> {
  static __device__ uint8_t convert(double v) {
    return dd::fp8e4m3fn_from_fp32_value(static_cast<float>(v));
  }
};
template <> struct FromF64<DType_::Float8E5M2, uint8_t> {
  static __device__ uint8_t convert(double v) {
    return dd::fp8e5m2_from_fp32_value(static_cast<float>(v));
  }
};
template <> struct FromF64<DType_::Signed8, int8_t> {
  static __device__ int8_t convert(double v) {
    if (isnan(v)) {
      return static_cast<int8_t>(0);
    }
    const double c = v < -128.0 ? -128.0 : (v > 127.0 ? 127.0 : v);
    return static_cast<int8_t>(c);
  }
};
template <> struct FromF64<DType_::UnSigned8, uint8_t> {
  static __device__ uint8_t convert(double v) {
    if (isnan(v)) {
      return static_cast<uint8_t>(0);
    }
    const double c = v <= 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v);
    return static_cast<uint8_t>(c);
  }
};
template <> struct FromF64<DType_::Signed16, int16_t> {
  static __device__ int16_t convert(double v) {
    if (isnan(v)) {
      return static_cast<int16_t>(0);
    }
    const double c = v < -32768.0 ? -32768.0 : (v > 32767.0 ? 32767.0 : v);
    return static_cast<int16_t>(c);
  }
};
template <> struct FromF64<DType_::UnSigned16, uint16_t> {
  static __device__ uint16_t convert(double v) {
    if (isnan(v)) {
      return static_cast<uint16_t>(0);
    }
    const double c = v <= 0.0 ? 0.0 : (v > 65535.0 ? 65535.0 : v);
    return static_cast<uint16_t>(c);
  }
};
template <> struct FromF64<DType_::Signed32, int32_t> {
  static __device__ int32_t convert(double v) {
    if (isnan(v)) {
      return static_cast<int32_t>(0);
    }
    const double c = v < -2147483648.0 ? -2147483648.0
                                       : (v > 2147483647.0 ? 2147483647.0 : v);
    return static_cast<int32_t>(c);
  }
};
template <> struct FromF64<DType_::UnSigned32, uint32_t> {
  static __device__ uint32_t convert(double v) {
    if (isnan(v)) {
      return static_cast<uint32_t>(0);
    }
    const double c = v <= 0.0 ? 0.0 : (v > 4294967295.0 ? 4294967295.0 : v);
    return static_cast<uint32_t>(c);
  }
};
template <> struct FromF64<DType_::Signed64, int64_t> {
  static __device__ int64_t convert(double v) {
    if (isnan(v)) {
      return static_cast<int64_t>(0);
    }
    const double c =
        v < -9223372036854775808.0
            ? -9223372036854775808.0
            : (v > 9223372036854774784.0 ? 9223372036854774784.0 : v);
    return static_cast<int64_t>(c);
  }
};
template <> struct FromF64<DType_::UnSigned64, uint64_t> {
  static __device__ uint64_t convert(double v) {
    if (isnan(v)) {
      return static_cast<uint64_t>(0);
    }
    const double c =
        v <= 0.0 ? 0.0
                 : (v > 18446744073709549568.0 ? 18446744073709549568.0 : v);
    return static_cast<uint64_t>(c);
  }
};

/**
 * @brief Integer to integer conversion.
 *
 * Only 8 bit destinations saturate.
 */
template <DType_ D, DType_ S, typename DT, typename ST>
__device__ inline DT intToInt(ST v) {
  if constexpr (D == S) {
    return v;
  } else if constexpr (D == DType_::Signed8) {
    if constexpr (isSignedInt(S)) {
      const int64_t x = static_cast<int64_t>(v);
      const int64_t c = x < -128 ? -128 : (x > 127 ? 127 : x);
      return static_cast<DT>(c);
    } else {
      return static_cast<DT>(v > static_cast<ST>(127) ? static_cast<ST>(127)
                                                      : v);
    }
  } else if constexpr (D == DType_::UnSigned8) {
    if constexpr (S == DType_::Signed8) {
      return static_cast<DT>(v);
    } else if constexpr (isSignedInt(S)) {
      const int64_t x = static_cast<int64_t>(v);
      const int64_t c = x <= 0 ? 0 : (x > 255 ? 255 : x);
      return static_cast<DT>(c);
    } else {
      return static_cast<DT>(v > static_cast<ST>(255) ? static_cast<ST>(255)
                                                      : v);
    }
  } else {
    return static_cast<DT>(v);
  }
}

// One element conversion selected by the dtype pair.
template <DType_ S, DType_ D, typename ST, typename DT>
__device__ inline DT convertElem(ST v) {
  if constexpr (isIntDType(S) && isIntDType(D)) {
    return intToInt<D, S, DT, ST>(v);
  } else if constexpr (S == DType_::Float64 || D == DType_::Float64) {
    const double d = ToF64<S, ST>::convert(v);
    return FromF64<D, DT>::convert(d);
  } else {
    const float f = ToF32<S, ST>::convert(v);
    return FromF32<D, DT>::convert(f);
  }
}

/// Bit reinterpretation between element values and loaded words, so the
/// vectorized paths never change the bits a scalar conversion would see.
template <typename T> __device__ inline uint64_t bitsOf(T v) {
  if constexpr (sizeof(T) == 8U) {
    if constexpr (std::is_same_v<T, double>) {
      return static_cast<uint64_t>(__double_as_longlong(v));
    } else {
      return static_cast<uint64_t>(v);
    }
  } else if constexpr (sizeof(T) == 4U) {
    if constexpr (std::is_same_v<T, float>) {
      return __float_as_uint(v);
    } else {
      return static_cast<uint32_t>(v);
    }
  } else if constexpr (sizeof(T) == 2U) {
    if constexpr (std::is_same_v<T, F16T>) {
      return __half_as_ushort(v);
    } else if constexpr (std::is_same_v<T, BF16T>) {
      return __bfloat16_as_ushort(v);
    } else {
      return static_cast<uint16_t>(v);
    }
  } else {
    return static_cast<uint8_t>(v);
  }
}

template <typename T> __device__ inline T fromBits(uint64_t bits) {
  if constexpr (sizeof(T) == 8U) {
    if constexpr (std::is_same_v<T, double>) {
      return __longlong_as_double(static_cast<long long>(bits));
    } else {
      return static_cast<T>(bits);
    }
  } else if constexpr (sizeof(T) == 4U) {
    if constexpr (std::is_same_v<T, float>) {
      return __uint_as_float(static_cast<uint32_t>(bits));
    } else {
      return static_cast<T>(static_cast<uint32_t>(bits));
    }
  } else if constexpr (sizeof(T) == 2U) {
    const uint16_t h = static_cast<uint16_t>(bits);
    if constexpr (std::is_same_v<T, F16T>) {
      return __ushort_as_half(h);
    } else if constexpr (std::is_same_v<T, BF16T>) {
      return __ushort_as_bfloat16(h);
    } else {
      return static_cast<T>(h);
    }
  } else {
    return static_cast<T>(static_cast<uint8_t>(bits));
  }
}

/** @brief Whole transactions for the vectorized paths.
 *
 * @details
 * One grid-stride iteration moves a full chunk through one unrolled
 * load, one unrolled conversion run and one unrolled store. loadChunk
 * and storeChunk each touch only 16B units on the side they own; the
 * callers pair them with the counterpart width (plain: both sides
 * 16B; unpack: 8B load with storeChunk stores; pack: loadChunk loads
 * with a kOut-byte store).
 */

/// @brief Pulls W elements from @p p through whole 16B transactions.
template <typename ST, uint32_t W>
__device__ inline void loadChunk(const unsigned char *restrict p, ST *vals) {
  constexpr uint32_t kWords = (W * static_cast<uint32_t>(sizeof(ST))) / 4U;
  static_assert(kWords % 4U == 0U, "chunks fill whole 16B loads");
  uint32_t w[kWords];
#pragma unroll
  for (uint32_t t = 0U; t < kWords; t += 4U) {
    const uint4 raw = *reinterpret_cast<const uint4 *>(p + ((t / 4U) * 16U));
    w[t + 0U] = raw.x;
    w[t + 1U] = raw.y;
    w[t + 2U] = raw.z;
    w[t + 3U] = raw.w;
  }
#pragma unroll
  for (uint32_t k = 0U; k < W; ++k) {
    uint64_t bits = 0U;
    if constexpr (sizeof(ST) == 1U) {
      bits = static_cast<uint64_t>((w[k >> 2U] >> ((k & 3U) * 8U)) & 0xFFU);
    } else if constexpr (sizeof(ST) == 2U) {
      bits = static_cast<uint64_t>((w[k >> 1U] >> ((k & 1U) * 16U)) & 0xFFFFU);
    } else if constexpr (sizeof(ST) == 4U) {
      bits = static_cast<uint64_t>(w[k]);
    } else {
      bits = static_cast<uint64_t>(w[k << 1U]) |
             (static_cast<uint64_t>(w[(k << 1U) + 1U]) << 32U);
    }
    vals[k] = fromBits<ST>(bits);
  }
}

/// @brief Pushes W elements to @p p through whole 16B transactions.
template <typename DT, uint32_t W>
__device__ inline void storeChunk(unsigned char *restrict p, const DT *vals) {
  constexpr uint32_t kWords = (W * static_cast<uint32_t>(sizeof(DT))) / 4U;
  static_assert(kWords % 4U == 0U, "chunks fill whole 16B stores");
  uint32_t w[kWords] = {};
#pragma unroll
  for (uint32_t k = 0U; k < W; ++k) {
    const uint64_t bits = bitsOf(vals[k]);
    if constexpr (sizeof(DT) == 1U) {
      w[k >> 2U] |= static_cast<uint32_t>(bits & 0xFFU) << ((k & 3U) * 8U);
    } else if constexpr (sizeof(DT) == 2U) {
      w[k >> 1U] |= static_cast<uint32_t>(bits & 0xFFFFU) << ((k & 1U) * 16U);
    } else if constexpr (sizeof(DT) == 4U) {
      w[k] = static_cast<uint32_t>(bits);
    } else {
      w[k << 1U] = static_cast<uint32_t>(bits);
      w[(k << 1U) + 1U] = static_cast<uint32_t>(bits >> 32U);
    }
  }
#pragma unroll
  for (uint32_t t = 0U; t < kWords; t += 4U) {
    const uint4 packed = make_uint4(w[t + 0U], w[t + 1U], w[t + 2U], w[t + 3U]);
    *reinterpret_cast<uint4 *>(p + ((t / 4U) * 16U)) = packed;
  }
}

/** @brief Casts one dtype pair element by element.
 *
 * @details
 * 16B vector chunks when both pointers prove a 16B-aligned run
 * (W = 16 / narrower item elements per chunk); a scalar coarsened
 * grid stride loop covers tails and misaligned pointers, so any grid
 * size and any pointer stay correct.
 */
template <DType_ S, DType_ D, typename ST, typename DT>
__global__ void castPlainKernel(const ST *restrict src, DT *restrict dst,
                                size_t n) {
  constexpr uint32_t kMin = sizeof(ST) < sizeof(DT) ? sizeof(ST) : sizeof(DT);
  constexpr uint32_t W = 16U / kMin;
  constexpr uint32_t kIn = W * static_cast<uint32_t>(sizeof(ST));
  constexpr uint32_t kOut = W * static_cast<uint32_t>(sizeof(DT));
  static_assert(kIn % 16U == 0U && kOut % 16U == 0U,
                "plain chunks fill whole transactions");

  const bool vec = (reinterpret_cast<uintptr_t>(src) % 16U == 0U) &&
                   (reinterpret_cast<uintptr_t>(dst) % 16U == 0U) &&
                   (n >= static_cast<size_t>(W));
  const size_t tid =
      static_cast<size_t>(threadIdx.x) +
      (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x));
  const size_t stride =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
  if (vec) {
    const size_t nVec = n / static_cast<size_t>(W);
    const auto *sb = reinterpret_cast<const unsigned char *>(src);
    auto *db = reinterpret_cast<unsigned char *>(dst);
    for (size_t c = tid; c < nVec; c += stride) {
      ST vals[W];
      loadChunk<ST, W>(sb + (c * kIn), vals);
      DT out[W];
#pragma unroll
      for (uint32_t k = 0U; k < W; ++k) {
        out[k] = convertElem<S, D, ST, DT>(vals[k]);
      }
      storeChunk<DT, W>(db + (c * kOut), out);
    }
    for (size_t i = (nVec * static_cast<size_t>(W)) + tid; i < n; i += stride) {
      dst[i] = convertElem<S, D, ST, DT>(src[i]);
    }
    return;
  }
  for (size_t base = tid * 4U; base < n; base += stride * 4U) {
#pragma unroll
    for (uint32_t k = 0U; k < 4U; ++k) {
      const size_t i = base + static_cast<size_t>(k);
      if (i < n) {
        dst[i] = convertElem<S, D, ST, DT>(src[i]);
      }
    }
  }
}

/** @brief Fans each source byte out to two elements.
 *
 * @details
 * The vectorized path expands eight bytes into sixteen elements
 * behind one 8B load and whole 16B stores; the scalar masked loop
 * covers tails and misaligned pointers.
 */
template <DType_ D, typename DT>
__global__ void castUnpackKernel(const uint8_t *restrict src, DT *restrict dst,
                                 size_t n) {
  constexpr uint32_t kIn = 8U;
  constexpr uint32_t W = 16U;
  constexpr uint32_t kOut = 16U * static_cast<uint32_t>(sizeof(DT));
  static_assert(kOut % 16U == 0U, "unpack chunks fill whole stores");

  const bool vec = (reinterpret_cast<uintptr_t>(src) % 8U == 0U) &&
                   (reinterpret_cast<uintptr_t>(dst) % 16U == 0U) && (n >= kIn);
  const size_t tid =
      static_cast<size_t>(threadIdx.x) +
      (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x));
  const size_t stride =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
  if (vec) {
    const size_t nVec = n / kIn;
    const auto *sb = reinterpret_cast<const unsigned char *>(src);
    auto *db = reinterpret_cast<unsigned char *>(dst);
    for (size_t c = tid; c < nVec; c += stride) {
      const uint2 raw = *reinterpret_cast<const uint2 *>(sb + (c * kIn));
      const uint32_t words[2] = {raw.x, raw.y};
      DT out[W];
#pragma unroll
      for (uint32_t e = 0U; e < W; ++e) {
        const uint32_t byteIdx = e >> 1U;
        const uint32_t shift = ((byteIdx & 3U) * 8U) + ((e & 1U) * 4U);
        const uint32_t nib = (words[byteIdx >> 2U] >> shift) & 0xFU;
        out[e] = FromF32<D, DT>::convert(
            dd::fp4e2m1fn_to_fp32_value(static_cast<uint8_t>(nib)));
      }
      storeChunk<DT, W>(db + (c * kOut), out);
    }
    for (size_t i = (nVec * kIn) + tid; i < n; i += stride) {
      const uint8_t byte = src[i];
      const float lo = dd::fp4e2m1fn_to_fp32_value(
          static_cast<uint8_t>(byte & UINT8_C(0xF)));
      const float hi = dd::fp4e2m1fn_to_fp32_value(
          static_cast<uint8_t>((byte >> 4U) & UINT8_C(0xF)));
      dst[i * 2U] = FromF32<D, DT>::convert(lo);
      dst[(i * 2U) + 1U] = FromF32<D, DT>::convert(hi);
    }
    return;
  }
  for (size_t base = tid * 4U; base < n; base += stride * 4U) {
#pragma unroll
    for (uint32_t k = 0U; k < 4U; ++k) {
      const size_t i = base + static_cast<size_t>(k);
      if (i < n) {
        const uint8_t byte = src[i];
        const float lo = dd::fp4e2m1fn_to_fp32_value(
            static_cast<uint8_t>(byte & UINT8_C(0xF)));
        const float hi = dd::fp4e2m1fn_to_fp32_value(
            static_cast<uint8_t>((byte >> 4U) & UINT8_C(0xF)));
        dst[i * 2U] = FromF32<D, DT>::convert(lo);
        dst[(i * 2U) + 1U] = FromF32<D, DT>::convert(hi);
      }
    }
  }
}

/** @brief Folds two elements into one storage byte.
 *
 * @details
 * The vectorized path folds sixteen elements into 8/S bytes behind
 * one 16B load and one narrow store; the scalar masked loop covers
 * tails and misaligned pointers.
 */
template <DType_ S, typename ST>
__global__ void castPackKernel(const ST *restrict src, uint8_t *restrict dst,
                               size_t n) {
  constexpr uint32_t kOut = 8U / static_cast<uint32_t>(sizeof(ST));
  constexpr uint32_t W = kOut * 2U;
  constexpr uint32_t kIn = 16U;

  const bool vec = (reinterpret_cast<uintptr_t>(src) % 16U == 0U) &&
                   (reinterpret_cast<uintptr_t>(dst) % kOut == 0U) &&
                   (n >= kOut);
  const size_t tid =
      static_cast<size_t>(threadIdx.x) +
      (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x));
  const size_t stride =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
  if (vec) {
    const size_t nVec = n / kOut;
    const auto *sb = reinterpret_cast<const unsigned char *>(src);
    auto *db = reinterpret_cast<unsigned char *>(dst);
    for (size_t c = tid; c < nVec; c += stride) {
      ST vals[W];
      loadChunk<ST, W>(sb + (c * kIn), vals);
      uint8_t nib[W];
#pragma unroll
      for (uint32_t k = 0U; k < W; ++k) {
        nib[k] = dd::fp4e2m1fn_from_fp32_value(ToF32<S, ST>::convert(vals[k]));
      }
      uint8_t bytes[kOut];
#pragma unroll
      for (uint32_t b = 0U; b < kOut; ++b) {
        bytes[b] =
            static_cast<uint8_t>((nib[(b << 1U) + 1U] << 4U) | nib[b << 1U]);
      }
      if constexpr (kOut == 8U) {
        const uint2 packed = {
            .x = static_cast<uint32_t>(bytes[0]) |
                 (static_cast<uint32_t>(bytes[1]) << 8U) |
                 (static_cast<uint32_t>(bytes[2]) << 16U) |
                 (static_cast<uint32_t>(bytes[3]) << 24U),
            .y = static_cast<uint32_t>(bytes[4]) |
                 (static_cast<uint32_t>(bytes[5]) << 8U) |
                 (static_cast<uint32_t>(bytes[6]) << 16U) |
                 (static_cast<uint32_t>(bytes[7]) << 24U),
        };
        *reinterpret_cast<uint2 *>(db + (c * kOut)) = packed;
      } else if constexpr (kOut == 4U) {
        const uint32_t packed = static_cast<uint32_t>(bytes[0]) |
                                (static_cast<uint32_t>(bytes[1]) << 8U) |
                                (static_cast<uint32_t>(bytes[2]) << 16U) |
                                (static_cast<uint32_t>(bytes[3]) << 24U);
        *reinterpret_cast<uint32_t *>(db + (c * kOut)) = packed;
      } else if constexpr (kOut == 2U) {
        const uint16_t packed =
            static_cast<uint16_t>(static_cast<uint16_t>(bytes[0]) |
                                  (static_cast<uint16_t>(bytes[1]) << 8U));
        *reinterpret_cast<uint16_t *>(db + (c * kOut)) = packed;
      } else {
        *(db + (c * kOut)) = bytes[0];
      }
    }
    for (size_t i = (nVec * kOut) + tid; i < n; i += stride) {
      const float lo = ToF32<S, ST>::convert(src[i * 2U]);
      const float hi = ToF32<S, ST>::convert(src[(i * 2U) + 1U]);
      const uint8_t loNib = dd::fp4e2m1fn_from_fp32_value(lo);
      const uint8_t hiNib = dd::fp4e2m1fn_from_fp32_value(hi);
      dst[i] = static_cast<uint8_t>((hiNib << 4U) | loNib);
    }
    return;
  }
  for (size_t base = tid * 4U; base < n; base += stride * 4U) {
#pragma unroll
    for (uint32_t k = 0U; k < 4U; ++k) {
      const size_t i = base + static_cast<size_t>(k);
      if (i < n) {
        const float lo = ToF32<S, ST>::convert(src[i * 2U]);
        const float hi = ToF32<S, ST>::convert(src[(i * 2U) + 1U]);
        const uint8_t loNib = dd::fp4e2m1fn_from_fp32_value(lo);
        const uint8_t hiNib = dd::fp4e2m1fn_from_fp32_value(hi);
        dst[i] = static_cast<uint8_t>((hiNib << 4U) | loNib);
      }
    }
  }
}

// Explicit kernel instantiations visible to the device pass.
template __global__ void
castPlainKernel<DType_::Float32, DType_::Float64, float, double>(
    const float *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::Float16, float, F16T>(
    const float *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::BFloat16, float, BF16T>(
    const float *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::Float8E4M3fn, float, uint8_t>(
    const float *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::Float8E5M2, float, uint8_t>(
    const float *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::Signed8, float, int8_t>(
    const float *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::UnSigned8, float, uint8_t>(
    const float *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::Signed16, float, int16_t>(
    const float *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::UnSigned16, float, uint16_t>(
    const float *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::Signed32, float, int32_t>(
    const float *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::UnSigned32, float, uint32_t>(
    const float *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::Signed64, float, int64_t>(
    const float *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::UnSigned64, float, uint64_t>(
    const float *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::Float32, double, float>(
    const double *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::Float16, double, F16T>(
    const double *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::BFloat16, double, BF16T>(
    const double *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::Float8E4M3fn, double, uint8_t>(
    const double *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::Float8E5M2, double, uint8_t>(
    const double *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::Signed8, double, int8_t>(
    const double *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::UnSigned8, double, uint8_t>(
    const double *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::Signed16, double, int16_t>(
    const double *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::UnSigned16, double, uint16_t>(
    const double *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::Signed32, double, int32_t>(
    const double *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::UnSigned32, double, uint32_t>(
    const double *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::Signed64, double, int64_t>(
    const double *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float64, DType_::UnSigned64, double, uint64_t>(
    const double *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::Float32, F16T, float>(
    const F16T *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::Float64, F16T, double>(
    const F16T *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::BFloat16, F16T, BF16T>(
    const F16T *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::Float8E4M3fn, F16T, uint8_t>(
    const F16T *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::Float8E5M2, F16T, uint8_t>(
    const F16T *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::Signed8, F16T, int8_t>(
    const F16T *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::UnSigned8, F16T, uint8_t>(
    const F16T *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::Signed16, F16T, int16_t>(
    const F16T *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::UnSigned16, F16T, uint16_t>(
    const F16T *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::Signed32, F16T, int32_t>(
    const F16T *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::UnSigned32, F16T, uint32_t>(
    const F16T *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::Signed64, F16T, int64_t>(
    const F16T *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float16, DType_::UnSigned64, F16T, uint64_t>(
    const F16T *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::Float32, BF16T, float>(
    const BF16T *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::Float64, BF16T, double>(
    const BF16T *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::Float16, BF16T, F16T>(
    const BF16T *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::Float8E4M3fn, BF16T, uint8_t>(
    const BF16T *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::Float8E5M2, BF16T, uint8_t>(
    const BF16T *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::Signed8, BF16T, int8_t>(
    const BF16T *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::UnSigned8, BF16T, uint8_t>(
    const BF16T *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::Signed16, BF16T, int16_t>(
    const BF16T *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::UnSigned16, BF16T, uint16_t>(
    const BF16T *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::Signed32, BF16T, int32_t>(
    const BF16T *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::UnSigned32, BF16T, uint32_t>(
    const BF16T *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::Signed64, BF16T, int64_t>(
    const BF16T *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::BFloat16, DType_::UnSigned64, BF16T, uint64_t>(
    const BF16T *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::Float32, uint8_t, float>(
    const uint8_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::Float64, uint8_t, double>(
    const uint8_t *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::Float16, uint8_t, F16T>(
    const uint8_t *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::BFloat16, uint8_t, BF16T>(
    const uint8_t *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::Float8E5M2, uint8_t, uint8_t>(
    const uint8_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::Signed8, uint8_t, int8_t>(
    const uint8_t *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::UnSigned8, uint8_t, uint8_t>(
    const uint8_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::Signed16, uint8_t, int16_t>(
    const uint8_t *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::UnSigned16, uint8_t, uint16_t>(
    const uint8_t *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::Signed32, uint8_t, int32_t>(
    const uint8_t *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::UnSigned32, uint8_t, uint32_t>(
    const uint8_t *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::Signed64, uint8_t, int64_t>(
    const uint8_t *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E4M3fn, DType_::UnSigned64, uint8_t, uint64_t>(
    const uint8_t *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::Float32, uint8_t, float>(
    const uint8_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::Float64, uint8_t, double>(
    const uint8_t *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::Float16, uint8_t, F16T>(
    const uint8_t *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::BFloat16, uint8_t, BF16T>(
    const uint8_t *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::Float8E4M3fn, uint8_t, uint8_t>(
    const uint8_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::Signed8, uint8_t, int8_t>(
    const uint8_t *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::UnSigned8, uint8_t, uint8_t>(
    const uint8_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::Signed16, uint8_t, int16_t>(
    const uint8_t *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::UnSigned16, uint8_t, uint16_t>(
    const uint8_t *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::Signed32, uint8_t, int32_t>(
    const uint8_t *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::UnSigned32, uint8_t, uint32_t>(
    const uint8_t *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::Signed64, uint8_t, int64_t>(
    const uint8_t *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E5M2, DType_::UnSigned64, uint8_t, uint64_t>(
    const uint8_t *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float32, DType_::Float8E8M0fnu, float, uint8_t>(
    const float *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Float8E8M0fnu, DType_::Float32, uint8_t, float>(
    const uint8_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::Float32, int8_t, float>(
    const int8_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::Float64, int8_t, double>(
    const int8_t *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::Float16, int8_t, F16T>(
    const int8_t *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::BFloat16, int8_t, BF16T>(
    const int8_t *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::Float8E4M3fn, int8_t, uint8_t>(
    const int8_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::Float8E5M2, int8_t, uint8_t>(
    const int8_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::UnSigned8, int8_t, uint8_t>(
    const int8_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::Signed16, int8_t, int16_t>(
    const int8_t *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::UnSigned16, int8_t, uint16_t>(
    const int8_t *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::Signed32, int8_t, int32_t>(
    const int8_t *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::UnSigned32, int8_t, uint32_t>(
    const int8_t *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::Signed64, int8_t, int64_t>(
    const int8_t *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed8, DType_::UnSigned64, int8_t, uint64_t>(
    const int8_t *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::Float32, uint8_t, float>(
    const uint8_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::Float64, uint8_t, double>(
    const uint8_t *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::Float16, uint8_t, F16T>(
    const uint8_t *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::BFloat16, uint8_t, BF16T>(
    const uint8_t *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::Float8E4M3fn, uint8_t, uint8_t>(
    const uint8_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::Float8E5M2, uint8_t, uint8_t>(
    const uint8_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::Signed8, uint8_t, int8_t>(
    const uint8_t *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::Signed16, uint8_t, int16_t>(
    const uint8_t *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::UnSigned16, uint8_t, uint16_t>(
    const uint8_t *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::Signed32, uint8_t, int32_t>(
    const uint8_t *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::UnSigned32, uint8_t, uint32_t>(
    const uint8_t *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::Signed64, uint8_t, int64_t>(
    const uint8_t *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned8, DType_::UnSigned64, uint8_t, uint64_t>(
    const uint8_t *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::Float32, int16_t, float>(
    const int16_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::Float64, int16_t, double>(
    const int16_t *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::Float16, int16_t, F16T>(
    const int16_t *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::BFloat16, int16_t, BF16T>(
    const int16_t *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::Float8E4M3fn, int16_t, uint8_t>(
    const int16_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::Float8E5M2, int16_t, uint8_t>(
    const int16_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::Signed8, int16_t, int8_t>(
    const int16_t *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::UnSigned8, int16_t, uint8_t>(
    const int16_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::UnSigned16, int16_t, uint16_t>(
    const int16_t *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::Signed32, int16_t, int32_t>(
    const int16_t *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::UnSigned32, int16_t, uint32_t>(
    const int16_t *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::Signed64, int16_t, int64_t>(
    const int16_t *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed16, DType_::UnSigned64, int16_t, uint64_t>(
    const int16_t *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::Float32, uint16_t, float>(
    const uint16_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::Float64, uint16_t, double>(
    const uint16_t *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::Float16, uint16_t, F16T>(
    const uint16_t *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::BFloat16, uint16_t, BF16T>(
    const uint16_t *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::Float8E4M3fn, uint16_t, uint8_t>(
    const uint16_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::Float8E5M2, uint16_t, uint8_t>(
    const uint16_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::Signed8, uint16_t, int8_t>(
    const uint16_t *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::UnSigned8, uint16_t, uint8_t>(
    const uint16_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::Signed16, uint16_t, int16_t>(
    const uint16_t *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::Signed32, uint16_t, int32_t>(
    const uint16_t *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::UnSigned32, uint16_t, uint32_t>(
    const uint16_t *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::Signed64, uint16_t, int64_t>(
    const uint16_t *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned16, DType_::UnSigned64, uint16_t, uint64_t>(
    const uint16_t *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::Float32, int32_t, float>(
    const int32_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::Float64, int32_t, double>(
    const int32_t *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::Float16, int32_t, F16T>(
    const int32_t *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::BFloat16, int32_t, BF16T>(
    const int32_t *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::Float8E4M3fn, int32_t, uint8_t>(
    const int32_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::Float8E5M2, int32_t, uint8_t>(
    const int32_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::Signed8, int32_t, int8_t>(
    const int32_t *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::UnSigned8, int32_t, uint8_t>(
    const int32_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::Signed16, int32_t, int16_t>(
    const int32_t *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::UnSigned16, int32_t, uint16_t>(
    const int32_t *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::UnSigned32, int32_t, uint32_t>(
    const int32_t *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::Signed64, int32_t, int64_t>(
    const int32_t *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed32, DType_::UnSigned64, int32_t, uint64_t>(
    const int32_t *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::Float32, uint32_t, float>(
    const uint32_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::Float64, uint32_t, double>(
    const uint32_t *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::Float16, uint32_t, F16T>(
    const uint32_t *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::BFloat16, uint32_t, BF16T>(
    const uint32_t *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::Float8E4M3fn, uint32_t, uint8_t>(
    const uint32_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::Float8E5M2, uint32_t, uint8_t>(
    const uint32_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::Signed8, uint32_t, int8_t>(
    const uint32_t *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::UnSigned8, uint32_t, uint8_t>(
    const uint32_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::Signed16, uint32_t, int16_t>(
    const uint32_t *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::UnSigned16, uint32_t, uint16_t>(
    const uint32_t *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::Signed32, uint32_t, int32_t>(
    const uint32_t *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::Signed64, uint32_t, int64_t>(
    const uint32_t *restrict, int64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned32, DType_::UnSigned64, uint32_t, uint64_t>(
    const uint32_t *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::Float32, int64_t, float>(
    const int64_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::Float64, int64_t, double>(
    const int64_t *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::Float16, int64_t, F16T>(
    const int64_t *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::BFloat16, int64_t, BF16T>(
    const int64_t *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::Float8E4M3fn, int64_t, uint8_t>(
    const int64_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::Float8E5M2, int64_t, uint8_t>(
    const int64_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::Signed8, int64_t, int8_t>(
    const int64_t *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::UnSigned8, int64_t, uint8_t>(
    const int64_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::Signed16, int64_t, int16_t>(
    const int64_t *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::UnSigned16, int64_t, uint16_t>(
    const int64_t *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::Signed32, int64_t, int32_t>(
    const int64_t *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::UnSigned32, int64_t, uint32_t>(
    const int64_t *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::Signed64, DType_::UnSigned64, int64_t, uint64_t>(
    const int64_t *restrict, uint64_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::Float32, uint64_t, float>(
    const uint64_t *restrict, float *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::Float64, uint64_t, double>(
    const uint64_t *restrict, double *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::Float16, uint64_t, F16T>(
    const uint64_t *restrict, F16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::BFloat16, uint64_t, BF16T>(
    const uint64_t *restrict, BF16T *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::Float8E4M3fn, uint64_t, uint8_t>(
    const uint64_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::Float8E5M2, uint64_t, uint8_t>(
    const uint64_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::Signed8, uint64_t, int8_t>(
    const uint64_t *restrict, int8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::UnSigned8, uint64_t, uint8_t>(
    const uint64_t *restrict, uint8_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::Signed16, uint64_t, int16_t>(
    const uint64_t *restrict, int16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::UnSigned16, uint64_t, uint16_t>(
    const uint64_t *restrict, uint16_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::Signed32, uint64_t, int32_t>(
    const uint64_t *restrict, int32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::UnSigned32, uint64_t, uint32_t>(
    const uint64_t *restrict, uint32_t *restrict, size_t);
template __global__ void
castPlainKernel<DType_::UnSigned64, DType_::Signed64, uint64_t, int64_t>(
    const uint64_t *restrict, int64_t *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::Float32, float>(const uint8_t *restrict,
                                         float *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::Float64, double>(const uint8_t *restrict,
                                          double *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::Float16, F16T>(const uint8_t *restrict, F16T *restrict,
                                        size_t);
template __global__ void
castUnpackKernel<DType_::BFloat16, BF16T>(const uint8_t *restrict,
                                          BF16T *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::Float8E4M3fn, uint8_t>(const uint8_t *restrict,
                                                uint8_t *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::Float8E5M2, uint8_t>(const uint8_t *restrict,
                                              uint8_t *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::Signed8, int8_t>(const uint8_t *restrict,
                                          int8_t *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::UnSigned8, uint8_t>(const uint8_t *restrict,
                                             uint8_t *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::Signed16, int16_t>(const uint8_t *restrict,
                                            int16_t *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::UnSigned16, uint16_t>(const uint8_t *restrict,
                                               uint16_t *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::Signed32, int32_t>(const uint8_t *restrict,
                                            int32_t *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::UnSigned32, uint32_t>(const uint8_t *restrict,
                                               uint32_t *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::Signed64, int64_t>(const uint8_t *restrict,
                                            int64_t *restrict, size_t);
template __global__ void
castUnpackKernel<DType_::UnSigned64, uint64_t>(const uint8_t *restrict,
                                               uint64_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::Float32, float>(const float *restrict, uint8_t *restrict,
                                       size_t);
template __global__ void
castPackKernel<DType_::Float64, double>(const double *restrict,
                                        uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::Float16, F16T>(const F16T *restrict, uint8_t *restrict,
                                      size_t);
template __global__ void
castPackKernel<DType_::BFloat16, BF16T>(const BF16T *restrict,
                                        uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::Float8E4M3fn, uint8_t>(const uint8_t *restrict,
                                              uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::Float8E5M2, uint8_t>(const uint8_t *restrict,
                                            uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::Signed8, int8_t>(const int8_t *restrict,
                                        uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::UnSigned8, uint8_t>(const uint8_t *restrict,
                                           uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::Signed16, int16_t>(const int16_t *restrict,
                                          uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::UnSigned16, uint16_t>(const uint16_t *restrict,
                                             uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::Signed32, int32_t>(const int32_t *restrict,
                                          uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::UnSigned32, uint32_t>(const uint32_t *restrict,
                                             uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::Signed64, int64_t>(const int64_t *restrict,
                                          uint8_t *restrict, size_t);
template __global__ void
castPackKernel<DType_::UnSigned64, uint64_t>(const uint64_t *restrict,
                                             uint8_t *restrict, size_t);

#ifndef __CUDA_ARCH__

inline uint32_t castKey(DType_ src, DType_ dst) {
  return (static_cast<uint32_t>(src) << 8U) | static_cast<uint32_t>(dst);
}

using Launcher = novaStatus_t (*)(const Tensor *restrict, Tensor *restrict,
                                  cudaStream_t);

/// Per pair launcher: shape check, launch sizing, kernel launch.

/// Compiled facts of one cast kernel, cached by function pointer.
inline hk::KernelAttrs kernelAttrsFor(const void *fn) {
  thread_local std::array<std::pair<const void *, hk::KernelAttrs>, 8> cache{};
  thread_local size_t n = 0;
  for (size_t i = 0; i < n; ++i) {
    if (cache[i].first == fn) {
      return cache[i].second;
    }
  }
  hk::KernelAttrs a{};
  cudaFuncAttributes q{};
  if (cudaFuncGetAttributes(&q, fn) == cudaSuccess) {
    a.regsPerThread = static_cast<uint32_t>(q.numRegs);
    a.staticSmemBytes = static_cast<size_t>(q.sharedSizeBytes);
    a.maxThreadsPerBlockKernel = static_cast<uint32_t>(q.maxThreadsPerBlock);
  }
  if (n < cache.size()) {
    cache[n].first = fn;
    cache[n].second = a;
    ++n;
  }
  return a;
}

template <DType_ S, DType_ D, typename ST, typename DT, CastKind K>
novaStatus_t launchCast(const Tensor *restrict src, Tensor *restrict dst,
                        cudaStream_t stream) {
  size_t n = 0U;
  if constexpr (K == CastKind::kPlain) {
    if (src->size != dst->size) {
      return {.err = novaShapeMismatch,
              .message = nova_get_error_msg(novaShapeMismatch, nullptr)};
    }
    n = src->size;
  } else if constexpr (K == CastKind::kUnpack) {
    if (dst->size != src->size * 2U) {
      return {.err = novaShapeMismatch,
              .message = nova_get_error_msg(novaShapeMismatch, nullptr)};
    }
    n = src->size;
  } else {
    if (src->size != dst->size * 2U) {
      return {.err = novaShapeMismatch,
              .message = nova_get_error_msg(novaShapeMismatch, nullptr)};
    }
    n = dst->size;
  }

  if (n == 0U) {
    return OK;
  }

  novaStatus_t status{};
  auto properties = getCudaDeviceProperties(&status);
  if (status.err != novaSuccess) {
    return status;
  }
  hk::ElementWiseParams hp{};
  hp.device = hk::capsFromDetected(properties);
  if constexpr (K == CastKind::kPlain) {
    hp.numElements = static_cast<uint64_t>(n);
    hp.packedElemsPerUnit = 1U;
  } else {
    hp.numElements = static_cast<uint64_t>(n * 2U);
    hp.packedElemsPerUnit = 2U;
  }
  hp.inputItemSize = static_cast<uint32_t>(src->item_size);
  hp.outputItemSize = static_cast<uint32_t>(dst->item_size);
  hp.inputAlignBytes = hk::pointerAlign(src->data.data + src->offset);
  hp.outputAlignBytes = hk::pointerAlign(dst->data.data);
  hp.inputContiguous = true;
  hp.outputContiguous = true;
  hp.numInputs = 1U;
  hp.arithmeticIntensity = 0.0F;
  hp.reuseAfter = false;
  const void *kernelPtr = nullptr;
  if constexpr (K == CastKind::kPlain) {
    kernelPtr = reinterpret_cast<const void *>(&castPlainKernel<S, D, ST, DT>);
  } else if constexpr (K == CastKind::kUnpack) {
    kernelPtr = reinterpret_cast<const void *>(&castUnpackKernel<D, DT>);
  } else {
    kernelPtr = reinterpret_cast<const void *>(&castPackKernel<S, ST>);
  }
  hp.kernel = kernelAttrsFor(kernelPtr);
  thread_local hk::FullConfigCache cache;
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(hp, &cache);

  const dim3 threads(cfg.threads.x, cfg.threads.y, cfg.threads.z);
  const dim3 grid(cfg.blocks.x, cfg.blocks.y, cfg.blocks.z);

  const auto *s = reinterpret_cast<const ST *>(src->data.data + src->offset);
  auto *d = reinterpret_cast<DT *>(dst->data.data);
  if constexpr (K == CastKind::kPlain) {
    castPlainKernel<S, D, ST, DT><<<grid, threads, 0, stream>>>(s, d, n);
  } else if constexpr (K == CastKind::kUnpack) {
    castUnpackKernel<D, DT><<<grid, threads, 0, stream>>>(s, d, n);
  } else {
    castPackKernel<S, ST><<<grid, threads, 0, stream>>>(s, d, n);
  }
  const cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    return {.err = novaKernelLaunchError,
            .message = cudaGetErrorString(launchErr)};
  }
  return OK;
}

const std::unordered_map<uint32_t, Launcher> &castMap() {
  static const std::unordered_map<uint32_t, Launcher> kMap = {
      {castKey(DType_::Float32, DType_::Float64),
       &launchCast<DType_::Float32, DType_::Float64, float, double,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::Float16),
       &launchCast<DType_::Float32, DType_::Float16, float, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::BFloat16),
       &launchCast<DType_::Float32, DType_::BFloat16, float, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::Float8E4M3fn),
       &launchCast<DType_::Float32, DType_::Float8E4M3fn, float, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::Float8E5M2),
       &launchCast<DType_::Float32, DType_::Float8E5M2, float, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::Signed8),
       &launchCast<DType_::Float32, DType_::Signed8, float, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::UnSigned8),
       &launchCast<DType_::Float32, DType_::UnSigned8, float, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::Signed16),
       &launchCast<DType_::Float32, DType_::Signed16, float, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::UnSigned16),
       &launchCast<DType_::Float32, DType_::UnSigned16, float, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::Signed32),
       &launchCast<DType_::Float32, DType_::Signed32, float, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::UnSigned32),
       &launchCast<DType_::Float32, DType_::UnSigned32, float, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::Signed64),
       &launchCast<DType_::Float32, DType_::Signed64, float, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::UnSigned64),
       &launchCast<DType_::Float32, DType_::UnSigned64, float, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::Float32),
       &launchCast<DType_::Float64, DType_::Float32, double, float,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::Float16),
       &launchCast<DType_::Float64, DType_::Float16, double, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::BFloat16),
       &launchCast<DType_::Float64, DType_::BFloat16, double, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::Float8E4M3fn),
       &launchCast<DType_::Float64, DType_::Float8E4M3fn, double, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::Float8E5M2),
       &launchCast<DType_::Float64, DType_::Float8E5M2, double, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::Signed8),
       &launchCast<DType_::Float64, DType_::Signed8, double, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::UnSigned8),
       &launchCast<DType_::Float64, DType_::UnSigned8, double, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::Signed16),
       &launchCast<DType_::Float64, DType_::Signed16, double, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::UnSigned16),
       &launchCast<DType_::Float64, DType_::UnSigned16, double, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::Signed32),
       &launchCast<DType_::Float64, DType_::Signed32, double, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::UnSigned32),
       &launchCast<DType_::Float64, DType_::UnSigned32, double, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::Signed64),
       &launchCast<DType_::Float64, DType_::Signed64, double, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float64, DType_::UnSigned64),
       &launchCast<DType_::Float64, DType_::UnSigned64, double, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::Float32),
       &launchCast<DType_::Float16, DType_::Float32, F16T, float,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::Float64),
       &launchCast<DType_::Float16, DType_::Float64, F16T, double,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::BFloat16),
       &launchCast<DType_::Float16, DType_::BFloat16, F16T, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::Float8E4M3fn),
       &launchCast<DType_::Float16, DType_::Float8E4M3fn, F16T, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::Float8E5M2),
       &launchCast<DType_::Float16, DType_::Float8E5M2, F16T, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::Signed8),
       &launchCast<DType_::Float16, DType_::Signed8, F16T, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::UnSigned8),
       &launchCast<DType_::Float16, DType_::UnSigned8, F16T, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::Signed16),
       &launchCast<DType_::Float16, DType_::Signed16, F16T, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::UnSigned16),
       &launchCast<DType_::Float16, DType_::UnSigned16, F16T, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::Signed32),
       &launchCast<DType_::Float16, DType_::Signed32, F16T, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::UnSigned32),
       &launchCast<DType_::Float16, DType_::UnSigned32, F16T, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::Signed64),
       &launchCast<DType_::Float16, DType_::Signed64, F16T, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float16, DType_::UnSigned64),
       &launchCast<DType_::Float16, DType_::UnSigned64, F16T, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::Float32),
       &launchCast<DType_::BFloat16, DType_::Float32, BF16T, float,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::Float64),
       &launchCast<DType_::BFloat16, DType_::Float64, BF16T, double,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::Float16),
       &launchCast<DType_::BFloat16, DType_::Float16, BF16T, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::Float8E4M3fn),
       &launchCast<DType_::BFloat16, DType_::Float8E4M3fn, BF16T, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::Float8E5M2),
       &launchCast<DType_::BFloat16, DType_::Float8E5M2, BF16T, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::Signed8),
       &launchCast<DType_::BFloat16, DType_::Signed8, BF16T, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::UnSigned8),
       &launchCast<DType_::BFloat16, DType_::UnSigned8, BF16T, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::Signed16),
       &launchCast<DType_::BFloat16, DType_::Signed16, BF16T, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::UnSigned16),
       &launchCast<DType_::BFloat16, DType_::UnSigned16, BF16T, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::Signed32),
       &launchCast<DType_::BFloat16, DType_::Signed32, BF16T, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::UnSigned32),
       &launchCast<DType_::BFloat16, DType_::UnSigned32, BF16T, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::Signed64),
       &launchCast<DType_::BFloat16, DType_::Signed64, BF16T, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::BFloat16, DType_::UnSigned64),
       &launchCast<DType_::BFloat16, DType_::UnSigned64, BF16T, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::Float32),
       &launchCast<DType_::Float8E4M3fn, DType_::Float32, uint8_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::Float64),
       &launchCast<DType_::Float8E4M3fn, DType_::Float64, uint8_t, double,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::Float16),
       &launchCast<DType_::Float8E4M3fn, DType_::Float16, uint8_t, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::BFloat16),
       &launchCast<DType_::Float8E4M3fn, DType_::BFloat16, uint8_t, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::Float8E5M2),
       &launchCast<DType_::Float8E4M3fn, DType_::Float8E5M2, uint8_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::Signed8),
       &launchCast<DType_::Float8E4M3fn, DType_::Signed8, uint8_t, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::UnSigned8),
       &launchCast<DType_::Float8E4M3fn, DType_::UnSigned8, uint8_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::Signed16),
       &launchCast<DType_::Float8E4M3fn, DType_::Signed16, uint8_t, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::UnSigned16),
       &launchCast<DType_::Float8E4M3fn, DType_::UnSigned16, uint8_t, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::Signed32),
       &launchCast<DType_::Float8E4M3fn, DType_::Signed32, uint8_t, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::UnSigned32),
       &launchCast<DType_::Float8E4M3fn, DType_::UnSigned32, uint8_t, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::Signed64),
       &launchCast<DType_::Float8E4M3fn, DType_::Signed64, uint8_t, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E4M3fn, DType_::UnSigned64),
       &launchCast<DType_::Float8E4M3fn, DType_::UnSigned64, uint8_t, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::Float32),
       &launchCast<DType_::Float8E5M2, DType_::Float32, uint8_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::Float64),
       &launchCast<DType_::Float8E5M2, DType_::Float64, uint8_t, double,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::Float16),
       &launchCast<DType_::Float8E5M2, DType_::Float16, uint8_t, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::BFloat16),
       &launchCast<DType_::Float8E5M2, DType_::BFloat16, uint8_t, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::Float8E4M3fn),
       &launchCast<DType_::Float8E5M2, DType_::Float8E4M3fn, uint8_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::Signed8),
       &launchCast<DType_::Float8E5M2, DType_::Signed8, uint8_t, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::UnSigned8),
       &launchCast<DType_::Float8E5M2, DType_::UnSigned8, uint8_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::Signed16),
       &launchCast<DType_::Float8E5M2, DType_::Signed16, uint8_t, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::UnSigned16),
       &launchCast<DType_::Float8E5M2, DType_::UnSigned16, uint8_t, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::Signed32),
       &launchCast<DType_::Float8E5M2, DType_::Signed32, uint8_t, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::UnSigned32),
       &launchCast<DType_::Float8E5M2, DType_::UnSigned32, uint8_t, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::Signed64),
       &launchCast<DType_::Float8E5M2, DType_::Signed64, uint8_t, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E5M2, DType_::UnSigned64),
       &launchCast<DType_::Float8E5M2, DType_::UnSigned64, uint8_t, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float32, DType_::Float8E8M0fnu),
       &launchCast<DType_::Float32, DType_::Float8E8M0fnu, float, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float8E8M0fnu, DType_::Float32),
       &launchCast<DType_::Float8E8M0fnu, DType_::Float32, uint8_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::Float32),
       &launchCast<DType_::Signed8, DType_::Float32, int8_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::Float64),
       &launchCast<DType_::Signed8, DType_::Float64, int8_t, double,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::Float16),
       &launchCast<DType_::Signed8, DType_::Float16, int8_t, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::BFloat16),
       &launchCast<DType_::Signed8, DType_::BFloat16, int8_t, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::Float8E4M3fn),
       &launchCast<DType_::Signed8, DType_::Float8E4M3fn, int8_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::Float8E5M2),
       &launchCast<DType_::Signed8, DType_::Float8E5M2, int8_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::UnSigned8),
       &launchCast<DType_::Signed8, DType_::UnSigned8, int8_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::Signed16),
       &launchCast<DType_::Signed8, DType_::Signed16, int8_t, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::UnSigned16),
       &launchCast<DType_::Signed8, DType_::UnSigned16, int8_t, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::Signed32),
       &launchCast<DType_::Signed8, DType_::Signed32, int8_t, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::UnSigned32),
       &launchCast<DType_::Signed8, DType_::UnSigned32, int8_t, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::Signed64),
       &launchCast<DType_::Signed8, DType_::Signed64, int8_t, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed8, DType_::UnSigned64),
       &launchCast<DType_::Signed8, DType_::UnSigned64, int8_t, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::Float32),
       &launchCast<DType_::UnSigned8, DType_::Float32, uint8_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::Float64),
       &launchCast<DType_::UnSigned8, DType_::Float64, uint8_t, double,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::Float16),
       &launchCast<DType_::UnSigned8, DType_::Float16, uint8_t, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::BFloat16),
       &launchCast<DType_::UnSigned8, DType_::BFloat16, uint8_t, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::Float8E4M3fn),
       &launchCast<DType_::UnSigned8, DType_::Float8E4M3fn, uint8_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::Float8E5M2),
       &launchCast<DType_::UnSigned8, DType_::Float8E5M2, uint8_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::Signed8),
       &launchCast<DType_::UnSigned8, DType_::Signed8, uint8_t, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::Signed16),
       &launchCast<DType_::UnSigned8, DType_::Signed16, uint8_t, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::UnSigned16),
       &launchCast<DType_::UnSigned8, DType_::UnSigned16, uint8_t, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::Signed32),
       &launchCast<DType_::UnSigned8, DType_::Signed32, uint8_t, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::UnSigned32),
       &launchCast<DType_::UnSigned8, DType_::UnSigned32, uint8_t, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::Signed64),
       &launchCast<DType_::UnSigned8, DType_::Signed64, uint8_t, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned8, DType_::UnSigned64),
       &launchCast<DType_::UnSigned8, DType_::UnSigned64, uint8_t, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::Float32),
       &launchCast<DType_::Signed16, DType_::Float32, int16_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::Float64),
       &launchCast<DType_::Signed16, DType_::Float64, int16_t, double,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::Float16),
       &launchCast<DType_::Signed16, DType_::Float16, int16_t, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::BFloat16),
       &launchCast<DType_::Signed16, DType_::BFloat16, int16_t, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::Float8E4M3fn),
       &launchCast<DType_::Signed16, DType_::Float8E4M3fn, int16_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::Float8E5M2),
       &launchCast<DType_::Signed16, DType_::Float8E5M2, int16_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::Signed8),
       &launchCast<DType_::Signed16, DType_::Signed8, int16_t, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::UnSigned8),
       &launchCast<DType_::Signed16, DType_::UnSigned8, int16_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::UnSigned16),
       &launchCast<DType_::Signed16, DType_::UnSigned16, int16_t, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::Signed32),
       &launchCast<DType_::Signed16, DType_::Signed32, int16_t, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::UnSigned32),
       &launchCast<DType_::Signed16, DType_::UnSigned32, int16_t, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::Signed64),
       &launchCast<DType_::Signed16, DType_::Signed64, int16_t, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed16, DType_::UnSigned64),
       &launchCast<DType_::Signed16, DType_::UnSigned64, int16_t, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::Float32),
       &launchCast<DType_::UnSigned16, DType_::Float32, uint16_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::Float64),
       &launchCast<DType_::UnSigned16, DType_::Float64, uint16_t, double,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::Float16),
       &launchCast<DType_::UnSigned16, DType_::Float16, uint16_t, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::BFloat16),
       &launchCast<DType_::UnSigned16, DType_::BFloat16, uint16_t, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::Float8E4M3fn),
       &launchCast<DType_::UnSigned16, DType_::Float8E4M3fn, uint16_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::Float8E5M2),
       &launchCast<DType_::UnSigned16, DType_::Float8E5M2, uint16_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::Signed8),
       &launchCast<DType_::UnSigned16, DType_::Signed8, uint16_t, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::UnSigned8),
       &launchCast<DType_::UnSigned16, DType_::UnSigned8, uint16_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::Signed16),
       &launchCast<DType_::UnSigned16, DType_::Signed16, uint16_t, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::Signed32),
       &launchCast<DType_::UnSigned16, DType_::Signed32, uint16_t, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::UnSigned32),
       &launchCast<DType_::UnSigned16, DType_::UnSigned32, uint16_t, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::Signed64),
       &launchCast<DType_::UnSigned16, DType_::Signed64, uint16_t, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned16, DType_::UnSigned64),
       &launchCast<DType_::UnSigned16, DType_::UnSigned64, uint16_t, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::Float32),
       &launchCast<DType_::Signed32, DType_::Float32, int32_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::Float64),
       &launchCast<DType_::Signed32, DType_::Float64, int32_t, double,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::Float16),
       &launchCast<DType_::Signed32, DType_::Float16, int32_t, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::BFloat16),
       &launchCast<DType_::Signed32, DType_::BFloat16, int32_t, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::Float8E4M3fn),
       &launchCast<DType_::Signed32, DType_::Float8E4M3fn, int32_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::Float8E5M2),
       &launchCast<DType_::Signed32, DType_::Float8E5M2, int32_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::Signed8),
       &launchCast<DType_::Signed32, DType_::Signed8, int32_t, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::UnSigned8),
       &launchCast<DType_::Signed32, DType_::UnSigned8, int32_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::Signed16),
       &launchCast<DType_::Signed32, DType_::Signed16, int32_t, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::UnSigned16),
       &launchCast<DType_::Signed32, DType_::UnSigned16, int32_t, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::UnSigned32),
       &launchCast<DType_::Signed32, DType_::UnSigned32, int32_t, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::Signed64),
       &launchCast<DType_::Signed32, DType_::Signed64, int32_t, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed32, DType_::UnSigned64),
       &launchCast<DType_::Signed32, DType_::UnSigned64, int32_t, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::Float32),
       &launchCast<DType_::UnSigned32, DType_::Float32, uint32_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::Float64),
       &launchCast<DType_::UnSigned32, DType_::Float64, uint32_t, double,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::Float16),
       &launchCast<DType_::UnSigned32, DType_::Float16, uint32_t, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::BFloat16),
       &launchCast<DType_::UnSigned32, DType_::BFloat16, uint32_t, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::Float8E4M3fn),
       &launchCast<DType_::UnSigned32, DType_::Float8E4M3fn, uint32_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::Float8E5M2),
       &launchCast<DType_::UnSigned32, DType_::Float8E5M2, uint32_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::Signed8),
       &launchCast<DType_::UnSigned32, DType_::Signed8, uint32_t, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::UnSigned8),
       &launchCast<DType_::UnSigned32, DType_::UnSigned8, uint32_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::Signed16),
       &launchCast<DType_::UnSigned32, DType_::Signed16, uint32_t, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::UnSigned16),
       &launchCast<DType_::UnSigned32, DType_::UnSigned16, uint32_t, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::Signed32),
       &launchCast<DType_::UnSigned32, DType_::Signed32, uint32_t, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::Signed64),
       &launchCast<DType_::UnSigned32, DType_::Signed64, uint32_t, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned32, DType_::UnSigned64),
       &launchCast<DType_::UnSigned32, DType_::UnSigned64, uint32_t, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::Float32),
       &launchCast<DType_::Signed64, DType_::Float32, int64_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::Float64),
       &launchCast<DType_::Signed64, DType_::Float64, int64_t, double,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::Float16),
       &launchCast<DType_::Signed64, DType_::Float16, int64_t, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::BFloat16),
       &launchCast<DType_::Signed64, DType_::BFloat16, int64_t, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::Float8E4M3fn),
       &launchCast<DType_::Signed64, DType_::Float8E4M3fn, int64_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::Float8E5M2),
       &launchCast<DType_::Signed64, DType_::Float8E5M2, int64_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::Signed8),
       &launchCast<DType_::Signed64, DType_::Signed8, int64_t, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::UnSigned8),
       &launchCast<DType_::Signed64, DType_::UnSigned8, int64_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::Signed16),
       &launchCast<DType_::Signed64, DType_::Signed16, int64_t, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::UnSigned16),
       &launchCast<DType_::Signed64, DType_::UnSigned16, int64_t, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::Signed32),
       &launchCast<DType_::Signed64, DType_::Signed32, int64_t, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::UnSigned32),
       &launchCast<DType_::Signed64, DType_::UnSigned32, int64_t, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::Signed64, DType_::UnSigned64),
       &launchCast<DType_::Signed64, DType_::UnSigned64, int64_t, uint64_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::Float32),
       &launchCast<DType_::UnSigned64, DType_::Float32, uint64_t, float,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::Float64),
       &launchCast<DType_::UnSigned64, DType_::Float64, uint64_t, double,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::Float16),
       &launchCast<DType_::UnSigned64, DType_::Float16, uint64_t, F16T,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::BFloat16),
       &launchCast<DType_::UnSigned64, DType_::BFloat16, uint64_t, BF16T,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::Float8E4M3fn),
       &launchCast<DType_::UnSigned64, DType_::Float8E4M3fn, uint64_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::Float8E5M2),
       &launchCast<DType_::UnSigned64, DType_::Float8E5M2, uint64_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::Signed8),
       &launchCast<DType_::UnSigned64, DType_::Signed8, uint64_t, int8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::UnSigned8),
       &launchCast<DType_::UnSigned64, DType_::UnSigned8, uint64_t, uint8_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::Signed16),
       &launchCast<DType_::UnSigned64, DType_::Signed16, uint64_t, int16_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::UnSigned16),
       &launchCast<DType_::UnSigned64, DType_::UnSigned16, uint64_t, uint16_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::Signed32),
       &launchCast<DType_::UnSigned64, DType_::Signed32, uint64_t, int32_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::UnSigned32),
       &launchCast<DType_::UnSigned64, DType_::UnSigned32, uint64_t, uint32_t,
                   CastKind::kPlain>},
      {castKey(DType_::UnSigned64, DType_::Signed64),
       &launchCast<DType_::UnSigned64, DType_::Signed64, uint64_t, int64_t,
                   CastKind::kPlain>},
      {castKey(DType_::Float4E2M1fn, DType_::Float32),
       &launchCast<DType_::Float4E2M1fn, DType_::Float32, uint8_t, float,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::Float64),
       &launchCast<DType_::Float4E2M1fn, DType_::Float64, uint8_t, double,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::Float16),
       &launchCast<DType_::Float4E2M1fn, DType_::Float16, uint8_t, F16T,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::BFloat16),
       &launchCast<DType_::Float4E2M1fn, DType_::BFloat16, uint8_t, BF16T,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::Float8E4M3fn),
       &launchCast<DType_::Float4E2M1fn, DType_::Float8E4M3fn, uint8_t, uint8_t,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::Float8E5M2),
       &launchCast<DType_::Float4E2M1fn, DType_::Float8E5M2, uint8_t, uint8_t,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::Signed8),
       &launchCast<DType_::Float4E2M1fn, DType_::Signed8, uint8_t, int8_t,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::UnSigned8),
       &launchCast<DType_::Float4E2M1fn, DType_::UnSigned8, uint8_t, uint8_t,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::Signed16),
       &launchCast<DType_::Float4E2M1fn, DType_::Signed16, uint8_t, int16_t,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::UnSigned16),
       &launchCast<DType_::Float4E2M1fn, DType_::UnSigned16, uint8_t, uint16_t,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::Signed32),
       &launchCast<DType_::Float4E2M1fn, DType_::Signed32, uint8_t, int32_t,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::UnSigned32),
       &launchCast<DType_::Float4E2M1fn, DType_::UnSigned32, uint8_t, uint32_t,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::Signed64),
       &launchCast<DType_::Float4E2M1fn, DType_::Signed64, uint8_t, int64_t,
                   CastKind::kUnpack>},
      {castKey(DType_::Float4E2M1fn, DType_::UnSigned64),
       &launchCast<DType_::Float4E2M1fn, DType_::UnSigned64, uint8_t, uint64_t,
                   CastKind::kUnpack>},
      {castKey(DType_::Float32, DType_::Float4E2M1fn),
       &launchCast<DType_::Float32, DType_::Float4E2M1fn, float, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::Float64, DType_::Float4E2M1fn),
       &launchCast<DType_::Float64, DType_::Float4E2M1fn, double, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::Float16, DType_::Float4E2M1fn),
       &launchCast<DType_::Float16, DType_::Float4E2M1fn, F16T, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::BFloat16, DType_::Float4E2M1fn),
       &launchCast<DType_::BFloat16, DType_::Float4E2M1fn, BF16T, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::Float8E4M3fn, DType_::Float4E2M1fn),
       &launchCast<DType_::Float8E4M3fn, DType_::Float4E2M1fn, uint8_t, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::Float8E5M2, DType_::Float4E2M1fn),
       &launchCast<DType_::Float8E5M2, DType_::Float4E2M1fn, uint8_t, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::Signed8, DType_::Float4E2M1fn),
       &launchCast<DType_::Signed8, DType_::Float4E2M1fn, int8_t, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::UnSigned8, DType_::Float4E2M1fn),
       &launchCast<DType_::UnSigned8, DType_::Float4E2M1fn, uint8_t, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::Signed16, DType_::Float4E2M1fn),
       &launchCast<DType_::Signed16, DType_::Float4E2M1fn, int16_t, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::UnSigned16, DType_::Float4E2M1fn),
       &launchCast<DType_::UnSigned16, DType_::Float4E2M1fn, uint16_t, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::Signed32, DType_::Float4E2M1fn),
       &launchCast<DType_::Signed32, DType_::Float4E2M1fn, int32_t, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::UnSigned32, DType_::Float4E2M1fn),
       &launchCast<DType_::UnSigned32, DType_::Float4E2M1fn, uint32_t, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::Signed64, DType_::Float4E2M1fn),
       &launchCast<DType_::Signed64, DType_::Float4E2M1fn, int64_t, uint8_t,
                   CastKind::kPack>},
      {castKey(DType_::UnSigned64, DType_::Float4E2M1fn),
       &launchCast<DType_::UnSigned64, DType_::Float4E2M1fn, uint64_t, uint8_t,
                   CastKind::kPack>},
  };
  return kMap;
}

#endif

} // namespace

#ifndef __CUDA_ARCH__
/**
 * @brief Launch a dtype casting kernel on the CUDA device.
 *
 * @details
 * Backend-specific entry point called by @ref launchDtypeCastingKernel.
 * Selects the conversion from the runtime dtype pair through a lookup
 * table covering all 212 supported pairs, sizes the grid with
 * @ref ncore::heuristics::kernels::resolveLaunchConfig() from the
 * detected device properties, then launches the respective kernel.
 * Grid-stride loops keep every grid size correct; the helper only
 * tunes how the work spreads across the machine.
 *
 * @param[in]  src  Source tensor.  Must reside in CUDA device memory
 *                  with a supported source dtype.
 * @param[in,out] dst  Destination tensor.  Must reside in CUDA device
 *                     memory, have the target dtype, and match
 *                     @p src in shape.
 *
 * @return @ref novaSuccess on success, or an error status if device
 *         properties cannot be queried.
 *
 * @pre  Both @p src and @p dst must be allocated on the CUDA device.
 * @pre  @p src and @p dst must have identical shapes.
 * @post On success, @p dst contains the casted elements.
 *
 * @see launchDtypeCastingKernel()  Device-agnostic dispatch entry point.
 */
novaStatus_t launchCudaDtypeCastingKernel(const Tensor *restrict src,
                                          Tensor *restrict dst) {
  const uint32_t key = castKey(src->dtype, dst->dtype);
  const auto &table = castMap();
  const auto found = table.find(key);
  if (found == table.end()) {
    return {.err = novaCastNotSupported,
            .message = nova_get_error_msg(novaCastNotSupported, nullptr)};
  }
  cudaStream_t stream = nullptr;
  const novaStatus_t status = CudaEngineStream::current(&stream);
  if (status.err != novaSuccess) {
    return status;
  }
  return found->second(src, dst, stream);
}

#endif
