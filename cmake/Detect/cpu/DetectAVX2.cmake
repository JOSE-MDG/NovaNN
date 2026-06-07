#[=======================================================================[.rst:
.. module:: DetectAVX2
   :synopsis: Detect AVX2 and AVX2-VNNI compiler support.

Tests base AVX2 support first.  If AVX2 is available, conditionally
detects AVX2-VNNI (integer dot-product via the AVXVNNI extension).
``HAS_AVX2_INT8`` is set as an alias for ``HAS_AVX2_VNNI``.

**Result variables:**

- ``HAS_AVX2``      — Set to ``1`` if AVX2 is supported.
- ``HAS_AVX2_VNNI`` — Set to ``1`` if AVX2-VNNI is supported (requires AVX2).
- ``HAS_AVX2_INT8`` — Alias for ``HAS_AVX2_VNNI``.

**Appended flags:**

- ``-mavx2``      is added to ``SIMD_FLAGS`` on AVX2 success.
- ``-mavxvnni``   is added on AVX2-VNNI success.

**Instruction sets tested:**

- AVX2:      ``_mm256_add_epi32`` (256-bit integer add).
- AVX2-VNNI: ``_mm256_dpbusd_epi32`` (unsigned/signed byte dot-product).
#]=======================================================================]

check_simd(HAS_AVX2 "-mavx2" "-mavx2" "
    #include <immintrin.h>
    int main() { __m256i a = _mm256_set1_epi32(1); (void)_mm256_add_epi32(a, a); return 0; }
")

if(HAS_AVX2)
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
