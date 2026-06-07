#[=======================================================================[.rst:
.. module:: DetectAVX
   :synopsis: Detect AVX, F16C, and FMA3 compiler support.

Tests base AVX support first.  If AVX is available, conditionally
detects F16C (half-float conversion) and FMA3 (fused multiply-add).
If AVX is not found, both sub-extensions are forced to ``0``.

**Result variables:**

- ``HAS_AVX``  — Set to ``1`` if AVX is supported.
- ``HAS_F16C`` — Set to ``1`` if F16C is supported (requires AVX).
- ``HAS_FMA3`` — Set to ``1`` if FMA3 is supported (requires AVX).

**Appended flags:**

- ``-mavx``  is added to ``SIMD_FLAGS`` on AVX success.
- ``-mf16c`` is added on F16C success.
- ``-mfma``  is added on FMA3 success.

**Instruction sets tested:**

- AVX:   ``_mm256_add_ps`` (256-bit single-precision add).
- F16C:  ``_mm_cvtph_ps`` (half-to-float conversion).
- FMA3:  ``_mm256_fmadd_ps`` (256-bit fused multiply-add).
#]=======================================================================]

check_simd(HAS_AVX "-mavx" "-mavx" "
    #include <immintrin.h>
    int main() { __m256 a = _mm256_set1_ps(1.0f); (void)_mm256_add_ps(a, a); return 0; }
")

if(HAS_AVX)
    check_simd(HAS_F16C "-mavx -mf16c" "-mf16c" "
        #include <immintrin.h>
        int main() { __m128i h = _mm_set1_epi16(0x3C00); (void)_mm_cvtph_ps(h); return 0; }
    ")
    check_simd(HAS_FMA3 "-mavx -mfma" "-mfma" "
        #include <immintrin.h>
        int main() { __m256 a = _mm256_set1_ps(1.0f); (void)_mm256_fmadd_ps(a, a, a); return 0; }
    ")
else()
    set(HAS_F16C 0)
    set(HAS_FMA3 0)
endif()
