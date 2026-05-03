#include <ncore/simd.h>
#include <threads.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <string.h>

typedef unsigned int uint;

static Capabilities_ Caps_;
static once_flag init_flag = ONCE_FLAG_INIT;

static inline void detect_cpu_capabilities_(Capabilities_ *__restrict caps) {
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

  caps->sse4_2_ = (ecx & (1 << 20)) != 0;
  caps->fma3_ = (ecx & (1 << 12)) != 0;
  caps->avx_ = (ecx & (1 << 28)) != 0;
  caps->f16c_ = (ecx & (1 << 29)) != 0;

#ifdef _WIN32
  __cpuidex(cpu_info, 7, 0);
  eax = (uint)cpu_info[0];
  ebx = (uint)cpu_info[1];
  ecx = (uint)cpu_info[2];
  edx = (uint)cpu_info[3];
#else
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif

  caps->avx2_ = (ebx & (1 << 5)) != 0;
  caps->avx512f_ = (ebx & (1 << 16)) != 0;
  caps->avx512_vnni_ = (ecx & (1 << 11)) != 0;
  caps->amx_bf16_ = (edx & (1 << 22)) != 0;
  caps->avx512_fp16_ = (edx & (1 << 23)) != 0;
  caps->amx_int8_ = (edx & (1 << 25)) != 0;

#ifdef _WIN32
  __cpuidex(cpu_info, 7, 1);
  eax = (uint)cpu_info[0];
  ebx = (uint)cpu_info[1];
  ecx = (uint)cpu_info[2];
  edx = (uint)cpu_info[3];
#else
  __cpuid_count(7, 1, eax, ebx, ecx, edx);
#endif

  caps->avx2_vnni_ = (eax & (1 << 4)) != 0;
  caps->avx512_bf16_ = (eax & (1 << 5)) != 0;
  caps->amx_fp16_ = (eax & (1 << 21)) != 0;
  caps->avx2_int8_ = (edx & (1 << 4)) != 0;

  caps->vnni_ = (bool)(caps->avx512_vnni_ || caps->avx2_vnni_);
}

static inline void init_once() { detect_cpu_capabilities_(&Caps_); }

const Capabilities_ *get_cpu_capabilities() {
  call_once(&init_flag, init_once);
  return &Caps_;
}
