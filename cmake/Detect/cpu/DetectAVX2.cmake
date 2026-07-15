#[=======================================================================[.rst:
DetectAVX2
----------

Detect AVX2 and AVX2-VNNI instruction set support.  Sets
``HAS_AVX2`` to ``1`` if the compiler can emit AVX2 intrinsics.
When AVX2 is available, the module also probes for ``HAS_AVX2_VNNI``.

.. note::
  Unconditionally ``0`` under MSVC. See ``CheckInstructionSupport.cmake``
  for why -- no logic change needed here, ``check_simd`` handles the
  MSVC guard internally.

Variables defined:

``HAS_AVX2``
  ``1`` if AVX2 is supported, ``0`` otherwise.

``HAS_AVX2_VNNI``
  ``1`` if AVX2-VNNI is supported (requires AVX2).  Set to ``0``
  when AVX2 is absent.

``HAS_AVX2_INT8``
  Alias for ``HAS_AVX2_VNNI``.  Set to ``1`` when VNNI is available.

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
