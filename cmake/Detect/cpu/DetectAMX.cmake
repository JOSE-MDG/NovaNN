#[=======================================================================[.rst:
DetectAMX
---------

Detect Intel AMX (Advanced Matrix Extensions) tile and sub-extension
support.  Sets ``HAS_AMX_TILE`` to ``1`` if the compiler can emit
AMX tile intrinsics.  When tile support is present, the module probes
for FP16, BF16, and INT8 extensions.

.. note::
  Unconditionally ``0`` under MSVC. See ``CheckInstructionSupport.cmake``
  for why -- no logic change needed here, ``check_simd`` handles the
  MSVC guard internally.

Variables defined:

``HAS_AMX_TILE``
  ``1`` if AMX tile operations are supported, ``0`` otherwise.

``HAS_AMX``
  Alias for ``HAS_AMX_TILE``.

``HAS_AMX_FP16``
  ``1`` if AMX-FP16 is supported.

``HAS_AMX_BF16``
  ``1`` if AMX-BF16 is supported.

``HAS_AMX_INT8``
  ``1`` if AMX-INT8 is supported.

All sub-extension variables are set to ``0`` when ``HAS_AMX_TILE`` is
absent.

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
