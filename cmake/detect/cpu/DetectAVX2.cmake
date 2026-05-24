#[[
    @file DetectAVX2.cmake
    @brief AVX2 family detection — AVX2 base, AVX2-VNNI, AVX2-INT8.

    Detection order:
      1. AVX2 (256-bit integer SIMD) — always attempted.
      2. AVX2-VNNI (integer dot-product via AVXVNNI) — only if AVX2 succeeded.
         On success, sets HAS_AVX2_INT8 = 1 as well (since VNNI implies INT8).

    Variables set:
      - HAS_AVX2      — 1 if AVX2 is supported (-mavx2 appended to SIMD_FLAGS)
      - HAS_AVX2_VNNI — 1 if AVX2 VNNI is supported (-mavxvnni appended)
      - HAS_AVX2_INT8 — same as HAS_AVX2_VNNI (alias); 0 if no AVX2

    Requires: CheckInstructionSupport.cmake (provides check_simd macro)
    See also : DetectSIMD.cmake — orchestrator that includes this module
]]
check_simd(HAS_AVX2 "-mavx2" "-mavx2" "
    #include <immintrin.h>
    int main() { __m256i a = _mm256_set1_epi32(1); (void)_mm256_add_epi32(a, a); return 0; }
")

if(HAS_AVX2)
    #[[
        @var HAS_AVX2_VNNI
        @brief AVX2-VNNI (integer dot-product via AVXVNNI extension)
        @requires HAS_AVX2
    ]]
    check_simd(HAS_AVX2_VNNI "-mavx2 -mavxvnni" "-mavxvnni" "
        #include <immintrin.h>
        int main() {
            __m256i a = _mm256_set1_epi8(1), b = _mm256_set1_epi8(1), c = _mm256_set1_epi32(0);
            (void)_mm256_dpbusd_epi32(c, a, b); return 0;
        }
    ")
    if(HAS_AVX2_VNNI)
        set(HAS_AVX2_INT8 1)
    else()
        set(HAS_AVX2_INT8 0)
    endif()
else()
    set(HAS_AVX2_VNNI 0)
    set(HAS_AVX2_INT8 0)
endif()
