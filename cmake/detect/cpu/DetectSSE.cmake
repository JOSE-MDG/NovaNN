#[[
    @file DetectSSE.cmake
    @brief SSE4.2 instruction-set detection.

    Probes the compiler for SSE4.2 support by compiling (and running) a snippet
    that uses the CRC32 intrinsic (_mm_crc32_u32).  On success:
      - Sets HAS_SSE4_2 = 1
      - Appends -msse4.2 to SIMD_FLAGS

    Variables set:
      - HAS_SSE4_2  — 1 if SSE4.2 is supported, 0 otherwise

    Requires: CheckInstructionSupport.cmake (provides check_simd macro)
    See also : DetectSIMD.cmake — orchestrator that includes this module
]]
check_simd(HAS_SSE4_2 "-msse4.2" "-msse4.2" "
    #include <nmmintrin.h>
    int main() { (void)_mm_crc32_u32(0, 1); return 0; }
")
