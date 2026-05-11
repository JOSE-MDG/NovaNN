/**
 * @file simd.c
 * @brief CPU SIMD capability detection.
 */

#include <ncore/simd.h>
#include <threads.h>

#ifdef _WIN64
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <string.h>

typedef unsigned int uint;

static Capabilities_ Caps_; ///< Global capabilities cache
static once_flag init_flag = ONCE_FLAG_INIT;

/**
 * @brief Detect CPU SIMD capabilities via CPUID.
 * @param caps Output structure to populate.
 *
 * Queries CPUID leaves 1, 7, and 7/1 to detect available SIMD features.
 * Sets all fields to false before detection.
 */
static inline void detect_cpu_capabilities_(Capabilities_ *restrict caps) {
  memset(caps, 0, sizeof(Capabilities_));

  uint eax, ebx, ecx, edx;

#ifdef _WIN32
  int cpu_info[4];
  __cpuid(cpu_info, 1);
  eax = (uint)cpu_info[0];
  ebx = (uint)cpu_info[1];
  ecx = (uint)cpu_info[2];
  edx = (uint)cpu_info[3];
#else
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return;
  }
#endif

  caps->sse4_2_ = (bool)((ecx & (1 << 20)) != 0);
  caps->fma3_ = (bool)((ecx & (1 << 12)) != 0);
  caps->avx_ = (bool)((ecx & (1 << 28)) != 0);
  caps->f16c_ = (bool)((ecx & (1 << 29)) != 0);

#ifdef _WIN64
  __cpuidex(cpu_info, 7, 0);
  eax = (uint)cpu_info[0];
  ebx = (uint)cpu_info[1];
  ecx = (uint)cpu_info[2];
  edx = (uint)cpu_info[3];
#else
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif

  caps->avx2_ = (bool)((ebx & (1 << 5)) != 0);
  caps->avx512f_ = (bool)((ebx & (1 << 16)) != 0);
  caps->avx512_dq_ = (bool)((ebx & (1 << 17)) != 0);
  caps->avx512_bw_ = (bool)((ebx & (1 << 30)) != 0);
  caps->avx512_vl_ = (bool)((ebx & (1 << 31)) != 0);
  caps->avx512_vnni_ = (bool)((ecx & (1 << 11)) != 0);
  caps->amx_bf16_ = (bool)((edx & (1 << 22)) != 0);
  caps->avx512_fp16_ = (bool)((edx & (1 << 23)) != 0);
  caps->amx_int8_ = (bool)((edx & (1 << 25)) != 0);

#ifdef _WIN64
  __cpuidex(cpu_info, 7, 1);
  eax = (uint)cpu_info[0];
  ebx = (uint)cpu_info[1];
  ecx = (uint)cpu_info[2];
  edx = (uint)cpu_info[3];
#else
  __cpuid_count(7, 1, eax, ebx, ecx, edx);
#endif

  caps->avx2_vnni_ = (bool)((eax & (1 << 4)) != 0);
  caps->avx512_bf16_ = (bool)((eax & (1 << 5)) != 0);
  caps->amx_fp16_ = (bool)((eax & (1 << 21)) != 0);
  caps->avx2_int8_ = (bool)((edx & (1 << 4)) != 0);

  caps->amx_ = (bool)(caps->amx_bf16_ || caps->amx_fp16_ || caps->amx_int8_);
  caps->vnni_ = (bool)(caps->avx512_vnni_ || caps->avx2_vnni_);
}

/**
 * @brief One-time initializer for thread-safe lazy initialization.
 */
static inline void init_once() { detect_cpu_capabilities_(&Caps_); }

/**
 * @brief Get CPU capabilities (thread-safe singleton).
 * @return Pointer to global Capabilities_ structure.
 */
const Capabilities_ *get_cpu_capabilities() {
  call_once(&init_flag, init_once);
  return &Caps_;
}
