#[=======================================================================[.rst:
DetectAVX512
------------

Detect AVX-512 family instruction set support.  Sets ``HAS_AVX512F``
to ``1`` if the compiler can emit AVX-512 Foundation intrinsics.
When AVX-512F is available, the module probes for additional extensions.

.. note::
  Under MSVC all variables in this module are unconditionally set to
  ``0``. NovaNN's AVX-512 kernels are written with ``[[gnu::target(...)]]``
  and use the ``_ph`` intrinsic family for FP16, neither of which
  ``cl.exe`` supports (see ``CheckInstructionSupport.cmake``). MSVC
  builds always use the scalar fallback kernels instead.

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
absent (this includes MSVC, unconditionally).

#]=======================================================================]

if(MSVC)
  # See module-level note above.
  set(HAS_AVX512F 0)
  set(HAS_AVX512_BW 0)
  set(HAS_AVX512_DQ 0)
  set(HAS_AVX512_VL 0)
  set(HAS_AVX512_VNNI 0)
  set(HAS_AVX512_FP16 0)
  set(HAS_AVX512_BF16 0)
else()
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
          int main() {
              __m256 a = _mm256_set1_ps(1.0f), b = _mm256_set1_ps(2.0f);
              __mmask8 m = _mm256_cmp_ps_mask(a, b, _CMP_LT_OQ);
              (void)_mm256_maskz_add_ps(m, a, b);
              return 0;
          }
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
endif()
