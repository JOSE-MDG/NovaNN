#[[
    @file DetectAVX.cmake
    @brief AVX family detection — base AVX, F16C, FMA3.

    Detection order:
      1. AVX (256-bit vectors) — always attempted.
      2. F16C (FP16 conversion) — only if AVX succeeded.
      3. FMA3 (fused multiply-add) — only if AVX succeeded.

    Variables set:
      - HAS_AVX  — 1 if AVX is supported (-mavx appended to SIMD_FLAGS)
      - HAS_F16C — 1 if F16C is supported (-mf16c appended; 0 if no AVX)
      - HAS_FMA3 — 1 if FMA3 is supported (-mfma appended; 0 if no AVX)

    Requires: CheckInstructionSupport.cmake (provides check_simd macro)
    See also : DetectSIMD.cmake — orchestrator that includes this module
]]
check_simd(HAS_AVX "-mavx" "
    #include <immintrin.h>
    int main() { __m256 a = _mm256_set1_ps(1.0f); (void)_mm256_add_ps(a, a); return 0; }
")

#[[
    @var HAS_F16C
    @brief F16C (FP16 conversion instructions)
    @requires HAS_AVX

    @var HAS_FMA3
    @brief FMA3 (fused multiply-add, 3-operand)
    @requires HAS_AVX
]]
if(HAS_AVX)
    check_simd(HAS_F16C "-mavx -mf16c" "
        #include <immintrin.h>
        int main() { __m128i h = _mm_set1_epi16(0x3C00); (void)_mm_cvtph_ps(h); return 0; }
    ")
    check_simd(HAS_FMA3 "-mavx -mfma" "
        #include <immintrin.h>
        int main() { __m256 a = _mm256_set1_ps(1.0f); (void)_mm256_fmadd_ps(a, a, a); return 0; }
    ")
else()
    set(HAS_F16C 0)
    set(HAS_FMA3 0)
endif()
