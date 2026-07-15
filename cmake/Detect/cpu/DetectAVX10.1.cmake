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
  This snippet is only ever compiled and run via :command:`check_simd`,
  which is a no-op under MSVC (see ``CheckInstructionSupport.cmake``).
  There is therefore no MSVC/Windows branch here -- NovaNN never probes
  AVX10 on MSVC builds.

Variables defined:

``HAS_AVX10_1``
  ``1`` if the host CPU supports AVX10.1, ``0`` otherwise.

#]=======================================================================]

check_simd(HAS_AVX10_1 "-mavx10.1" "-mavx10.1" "
    #include <cpuid.h>

    int main() {
      unsigned int eax, ebx, ecx, edx;
      if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if (edx & (1U << 19)) {
          unsigned int eax24, ebx24, ecx24, edx24;
          __cpuid_count(0x24, 0, eax24, ebx24, ecx24, edx24);
          return (ebx24 & 0xFFU) >= 1 ? 0 : 1;
        }
      }
      return 1;
    }
")
