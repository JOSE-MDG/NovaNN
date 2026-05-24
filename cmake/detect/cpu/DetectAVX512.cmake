#[[
    @file DetectAVX512.cmake
    @brief AVX-512 family detection — F, BW, DQ, VL, VNNI, FP16, BF16.

    Detection order:
      1. AVX-512 Foundation (512-bit vectors) — always attempted; all
         subsequent checks depend on it.
      2. AVX-512 BW (byte/word)
      3. AVX-512 DQ (doubleword/quadword)
      4. AVX-512 VL (vector length — 128/256-bit with masking)
      5. AVX-512 VNNI (integer dot-product)
      6. AVX-512 FP16 (half-precision)
      7. AVX-512 BF16 (bfloat16)

    Variables set (all 0 if AVX-512F is unsupported):
      - HAS_AVX512F    — Foundation
      - HAS_AVX512_BW  — Byte/word
      - HAS_AVX512_DQ  — Doubleword/quadword
      - HAS_AVX512_VL  — Vector length
      - HAS_AVX512_VNNI — VNNI
      - HAS_AVX512_FP16 — FP16
      - HAS_AVX512_BF16 — BF16

    Requires: CheckInstructionSupport.cmake (provides check_simd macro)
    See also : DetectSIMD.cmake — orchestrator that includes this module
]]
check_simd(HAS_AVX512F "-mavx512f" "-mavx512f" "
    #include <immintrin.h>
    int main() { __m512 a = _mm512_set1_ps(1.0f); (void)_mm512_add_ps(a, a); return 0; }
")

if(HAS_AVX512F)
    check_simd(HAS_AVX512_BW "-mavx512f -mavx512bw" "-mavx512bw" "
        #include <immintrin.h>
        int main() { __m512i a = _mm512_set1_epi8(1); (void)_mm512_add_epi8(a, a); return 0; }
    ")
    check_simd(HAS_AVX512_DQ "-mavx512f -mavx512dq" "-mavx512dq" "
        #include <immintrin.h>
        int main() { __m512i a = _mm512_set1_epi64(1); (void)_mm512_mullo_epi64(a, a); return 0; }
    ")
    check_simd(HAS_AVX512_VL "-mavx512f -mavx512vl" "-mavx512vl" "
        #include <immintrin.h>
        int main() { __m256i a = _mm256_set1_epi32(1); __mmask8 m = _mm256_movepi32_mask(a); (void)m; return 0; }
    ")
    check_simd(HAS_AVX512_VNNI "-mavx512f -mavx512vnni" "-mavx512vnni" "
        #include <immintrin.h>
        int main() {
            __m512i a = _mm512_set1_epi8(1), b = _mm512_set1_epi8(1), c = _mm512_set1_epi32(0);
            (void)_mm512_dpbusd_epi32(c, a, b); return 0;
        }
    ")
    check_simd(HAS_AVX512_FP16 "-mavx512f -mavx512fp16" "-mavx512fp16" "
        #include <immintrin.h>
        int main() { __m512h a = _mm512_set1_ph(1.0f); (void)_mm512_add_ph(a, a); return 0; }
    ")
    check_simd(HAS_AVX512_BF16 "-mavx512f -mavx512bf16" "-mavx512bf16" "
        #include <immintrin.h>
        int main() { __m512 a = _mm512_set1_ps(1.0f); (void)_mm512_cvtne2ps_pbh(a, a); return 0; }
    ")
else()
    set(HAS_AVX512_BW 0)
    set(HAS_AVX512_DQ 0)
    set(HAS_AVX512_VL 0)
    set(HAS_AVX512_VNNI 0)
    set(HAS_AVX512_FP16 0)
    set(HAS_AVX512_BF16 0)
endif()
