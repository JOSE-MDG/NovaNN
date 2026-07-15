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
  This snippet is only ever compiled and run via :command:`check_simd`,
  which is a no-op under MSVC (see ``CheckInstructionSupport.cmake``).
  There is therefore no MSVC/Windows branch here -- NovaNN never probes
  AVX10 on MSVC builds.

Variables defined:

``HAS_AVX10_2``
  ``1`` if the host CPU supports AVX10.2, ``0`` otherwise.  Also ``0``
  when ``HAS_AVX10_1`` is absent.

#]=======================================================================]

if(HAS_AVX10_1)
  check_simd(HAS_AVX10_2 "-mavx10.2" "-mavx10.2" "
        #include <cpuid.h>

        int main() {
          unsigned int eax, ebx, ecx, edx;
          if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            if (edx & (1U << 19)) {
              unsigned int eax24, ebx24, ecx24, edx24;
              __cpuid_count(0x24, 0, eax24, ebx24, ecx24, edx24);
              return (ebx24 & 0xFFU) >= 2 ? 0 : 1;
            }
          }
          return 1;
        }
    ")
else()
  set(HAS_AVX10_2 0)
endif()
