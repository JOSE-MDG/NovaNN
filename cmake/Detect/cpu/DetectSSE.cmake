#[=======================================================================[.rst:
DetectSSE
---------

Detect SSE 4.2 instruction set support.  Sets ``HAS_SSE4_2`` to ``1``
if the compiler can emit and execute SSE 4.2 intrinsics.

.. note::
  Detection is performed via :command:`check_simd`, which wraps
  ``-m`` flags with ``/clang:`` prefix when using clang-cl.

#]=======================================================================]

check_simd(HAS_SSE4_2 "-msse4.2" "-msse4.2" "
    #include <nmmintrin.h>
    int main() { (void)_mm_crc32_u32(0, 1); return 0; }
")
