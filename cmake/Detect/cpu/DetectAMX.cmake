#[=======================================================================[.rst:
.. module:: DetectAMX
   :synopsis: Detect Intel AMX and sub-extension compiler support.

Tests base AMX-TILE support first.  If AMX-TILE is available,
sets ``HAS_AMX`` to ``1`` and conditionally detects three sub-extensions
for matrix operations.  If AMX-TILE is not found, all variables are
forced to ``0``.

**Result variables:**

- ``HAS_AMX``       — Set to ``1`` if AMX-TILE is supported.
- ``HAS_AMX_FP16``  — FP16 matrix multiply (requires AMX-TILE).
- ``HAS_AMX_BF16``  — BF16 matrix multiply (requires AMX-TILE).
- ``HAS_AMX_INT8``  — INT8 matrix multiply (requires AMX-TILE).

**Appended flags:**

- ``-mamx-tile``  on AMX-TILE success.
- ``-mamx-fp16``  on AMX-FP16 success.
- ``-mamx-bf16``  on AMX-BF16 success.
- ``-mamx-int8``  on AMX-INT8 success.

**Instruction sets tested:**

- AMX-TILE:  ``_tile_release`` (tile management).
- AMX-FP16:  ``_tile_dpfp16ps`` (FP16 matrix multiply-accumulate).
- AMX-BF16:  ``_tile_dpbf16ps`` (BF16 matrix multiply-accumulate).
- AMX-INT8:  ``_tile_dpbusd`` (INT8 matrix multiply-accumulate).

.. note::

   AMX requires ``_tile_release()`` calls around tile operations to
   avoid kernel conflicts.  All test snippets include these guards.
#]=======================================================================]

check_simd(HAS_AMX_TILE "-mamx-tile" "-mamx-tile" "
    #include <immintrin.h>
    int main() { _tile_release(); return 0; }
")

if(HAS_AMX_TILE)
    set(HAS_AMX 1)
    check_simd(HAS_AMX_FP16 "-mamx-tile -mamx-fp16" "-mamx-fp16" "
        #include <immintrin.h>
        int main() { _tile_release(); _tile_dpfp16ps(0, 1, 2); _tile_release(); return 0; }
    ")
    check_simd(HAS_AMX_BF16 "-mamx-tile -mamx-bf16" "-mamx-bf16" "
        #include <immintrin.h>
        int main() { _tile_release(); _tile_dpbf16ps(0, 1, 2); _tile_release(); return 0; }
    ")
    check_simd(HAS_AMX_INT8 "-mamx-tile -mamx-int8" "-mamx-int8" "
        #include <immintrin.h>
        int main() { _tile_release(); _tile_dpbusd(0, 1, 2); _tile_release(); return 0; }
    ")
else()
    set(HAS_AMX 0)
    set(HAS_AMX_FP16 0)
    set(HAS_AMX_BF16 0)
    set(HAS_AMX_INT8 0)
endif()
