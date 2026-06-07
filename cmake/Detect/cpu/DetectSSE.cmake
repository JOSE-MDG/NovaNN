#[=======================================================================[.rst:
.. module:: DetectSSE
   :synopsis: Detect SSE4.2 compiler support.

Tests whether the compiler can compile and run SSE4.2 intrinsics.
The test snippet exercises ``_mm_crc32_u32`` from ``<nmmintrin.h>``.

**Result variables:**

- ``HAS_SSE4_2`` — Set to ``1`` if SSE4.2 is supported.

**Appended flags:**

- ``-msse4.2`` is added to ``SIMD_FLAGS`` on success.
#]=======================================================================]

check_simd(HAS_SSE4_2 "-msse4.2" "-msse4.2" "
    #include <nmmintrin.h>
    int main() { (void)_mm_crc32_u32(0, 1); return 0; }
")
