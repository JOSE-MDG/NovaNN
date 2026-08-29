#[=======================================================================[.rst:
DetectAVX10.1
-------------

Detect AVX10.1 converged vector ISA support.  AVX10.1 is Intel's unified
vector ISA baseline that consolidates AVX-512 features under a single
CPUID flag -- it does not introduce new instructions beyond those already
exposed by AVX512F/BW/DQ/VL/VNNI.

Detection is performed by executing the CPUID instruction at configure
time (leaf 0x24, subleaf 0, EBX[7:0] >= 1), gated behind CPUID leaf 7,
subleaf 0, EDX[19] (AVX10 presence flag) -- leaf 0x24 is only
architecturally valid when that bit is set. Compiler-flag-based probing
is unreliable for AVX10 because the ISA defines no unique instruction
mnemonic to test.

.. note::
  This snippet is compiled and run via :command:`check_simd`, which
  wraps ``-m`` flags with ``/clang:`` prefix when using clang-cl.

Variables defined:

``HAS_AVX10_1``
  ``1`` if the host CPU supports AVX10.1, ``0`` otherwise.

#]=======================================================================]

check_simd(HAS_AVX10_1 "-mavx10.1" "-mavx10.1" "
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
        return (cpuid_avx10(&ebx) && (ebx & 0xFF) >= 1) ? 0 : 1;
    }
")
