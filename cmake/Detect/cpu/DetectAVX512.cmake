#[=======================================================================[.rst:
DetectAVX512
------------

Detect AVX-512 family instruction set support.  Sets ``HAS_AVX512F``
to ``1`` if the compiler can emit AVX-512 Foundation intrinsics.
When AVX-512F is available, the module probes for additional extensions.

Variables defined:

``HAS_AVX512F``
  ``1`` if AVX-512F is supported, ``0`` otherwise.

``HAS_AVX512_BW``
  ``1`` if AVX-512BW (byte/word) is supported.

``HAS_AVX512_DQ``
  ``1`` if AVX-512DQ (doubleword/quadword) is supported.

``HAS_AVX512_VL``
  ``1`` if AVX-512VL (vector length extensions) is supported.

``HAS_AVX512_VNNI``
  ``1`` if AVX-512VNNI is supported.

``HAS_AVX512_FP16``
  ``1`` if AVX-512FP16 is supported.

``HAS_AVX512_BF16``
  ``1`` if AVX-512BF16 is supported.

All sub-extension variables are set to ``0`` when ``HAS_AVX512F`` is
absent.

#]=======================================================================]

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
