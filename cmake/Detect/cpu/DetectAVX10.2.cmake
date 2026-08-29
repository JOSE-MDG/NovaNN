#[=======================================================================[.rst:
DetectAVX10.2
-------------

Detect AVX10.2 converged vector ISA support.  AVX10.2 extends AVX10.1
with new VNNI INT8/FP variants, MOVRS, and media/AI-oriented instructions
but, like AVX10.1, is identified by a CPUID version-number check rather
than by testing a unique instruction.

Requires ``HAS_AVX10_1`` -- this module only probes when AVX10.1 was
already detected.  Detection uses CPUID leaf 0x24, subleaf 0, EBX[7:0]
>= 2, gated behind CPUID leaf 7, subleaf 0, EDX[19] (AVX10 presence
flag), same as ``DetectAVX10.1``.

.. note::
  This snippet is compiled and run via :command:`check_simd`, which
  wraps ``-m`` flags with ``/clang:`` prefix when using clang-cl.

Variables defined:

``HAS_AVX10_2``
  ``1`` if the host CPU supports AVX10.2, ``0`` otherwise.  Also ``0``
  when ``HAS_AVX10_1`` is absent.

#]=======================================================================]

if(HAS_AVX10_1)
  check_simd(HAS_AVX10_2 "-mavx10.2" "-mavx10.2" "
#ifdef _WIN32
    #include <intrin.h>
    static int cpuid_avx10(int *ebx_out) {
        int regs[4];
        __cpuidex(regs, 7, 0);
        if (!(regs[3] & (1 << 19))) return 0;
        __cpuidex(regs, 0x24, 0);
        *ebx_out = regs[1];
        return 1;
    }
#else
    #include <cpuid.h>
    static int cpuid_avx10(int *ebx_out) {
        unsigned int eax, ebx, ecx, edx;
        if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
        if (!(edx & (1U << 19))) return 0;
        __cpuid_count(0x24, 0, eax, ebx, ecx, edx);
        *ebx_out = (int)ebx;
        return 1;
    }
#endif
    int main() {
        int ebx = 0;
        return (cpuid_avx10(&ebx) && (ebx & 0xFF) >= 2) ? 0 : 1;
    }
  ")
else()
  set(HAS_AVX10_2 0)
endif()
