#[=======================================================================[.rst:
DetectAVX
---------

Detect AVX, F16C, and FMA3 instruction set support.  Sets ``HAS_AVX``
to ``1`` if the compiler can emit AVX intrinsics.  When AVX is
available, the module also probes for ``HAS_F16C`` and ``HAS_FMA3``.

Variables defined:

``HAS_AVX``
  ``1`` if AVX is supported, ``0`` otherwise.

``HAS_F16C``
  ``1`` if F16C is supported (requires AVX).  Set to ``0`` when AVX
  is absent.

``HAS_FMA3``
  ``1`` if FMA3 is supported (requires AVX).  Set to ``0`` when AVX
  is absent.

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
