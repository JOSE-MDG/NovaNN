#[=======================================================================[.rst:
.. module:: DetectAVX512
   :synopsis: Detect AVX-512 and sub-extension compiler support.

Tests base AVX-512F support first.  If AVX-512F is available,
sequentially detects six sub-extensions.  If AVX-512F is not found,
all sub-extensions are forced to ``0``.

**Result variables:**

- ``HAS_AVX512F``    — Set to ``1`` if AVX-512F is supported.
- ``HAS_AVX512_BW``  — Byte/word operations (requires AVX-512F).
- ``HAS_AVX512_DQ``  — Doubleword/quadword operations (requires AVX-512F).
- ``HAS_AVX512_VL``  — Vector-length extensions (requires AVX-512F).
- ``HAS_AVX512_VNNI`` — Integer dot-product (requires AVX-512F).
- ``HAS_AVX512_FP16`` — Half-precision floating point (requires AVX-512F).
- ``HAS_AVX512_BF16`` — BFloat16 conversion (requires AVX-512F).

**Appended flags:**

- ``-mavx512f``     on AVX-512F success.
- ``-mavx512bw``    on BW success.
- ``-mavx512dq``    on DQ success.
- ``-mavx512vl``    on VL success.
- ``-mavx512vnni``  on VNNI success.
- ``-mavx512fp16``  on FP16 success.
- ``-mavx512bf16``  on BF16 success.

**Instruction sets tested:**

- AVX-512F:   ``_mm512_add_ps`` (512-bit float add).
- AVX-512BW:  ``_mm512_add_epi8`` (512-bit byte add).
- AVX-512DQ:  ``_mm512_mullo_epi64`` (64-bit integer multiply).
- AVX-512VL:  ``_mm256_movepi32_mask`` (256-bit mask move).
- AVX-512VNNI: ``_mm512_dpbusd_epi32`` (byte dot-product).
- AVX-512FP16: ``_mm512_add_ph`` (512-bit half-precision add).
- AVX-512BF16: ``_mm512_cvtne2ps_pbh`` (BF16 conversion).
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
