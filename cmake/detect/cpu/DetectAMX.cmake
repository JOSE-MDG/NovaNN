#[[
    @file DetectAMX.cmake
    @brief AMX family detection — Tile, FP16, BF16, INT8.

    Detection order:
      1. AMX-Tile (advanced matrix extensions base) — always attempted;
         all subsequent checks depend on it.
      2. AMX-FP16 (FP16 matrix multiply)
      3. AMX-BF16 (BF16 matrix multiply)
      4. AMX-INT8 (INT8 matrix multiply)

    Variables set (all 0 if AMX-Tile is unsupported):
      - HAS_AMX_TILE  — 1 if AMX-Tile is supported
      - HAS_AMX       — same as HAS_AMX_TILE (aggregate flag)
      - HAS_AMX_FP16  — 1 if AMX-FP16 is supported
      - HAS_AMX_BF16  — 1 if AMX-BF16 is supported
      - HAS_AMX_INT8  — 1 if AMX-INT8 is supported

    Requires: CheckInstructionSupport.cmake (provides check_simd macro)
    See also : DetectSIMD.cmake — orchestrator that includes this module
]]
check_simd(HAS_AMX_TILE "-mamx-tile" "
    #include <immintrin.h>
    int main() { _tile_release(); return 0; }
")

if(HAS_AMX_TILE)
    set(HAS_AMX 1)
    check_simd(HAS_AMX_FP16 "-mamx-tile -mamx-fp16" "
        #include <immintrin.h>
        int main() { _tile_release(); _tile_dpfp16ps(0, 1, 2); _tile_release(); return 0; }
    ")
    check_simd(HAS_AMX_BF16 "-mamx-tile -mamx-bf16" "
        #include <immintrin.h>
        int main() { _tile_release(); _tile_dpbf16ps(0, 1, 2); _tile_release(); return 0; }
    ")
    check_simd(HAS_AMX_INT8 "-mamx-tile -mamx-int8" "
        #include <immintrin.h>
        int main() { _tile_release(); _tile_dpbusd(0, 1, 2); _tile_release(); return 0; }
    ")
else()
    set(HAS_AMX 0)
    set(HAS_AMX_FP16 0)
    set(HAS_AMX_BF16 0)
    set(HAS_AMX_INT8 0)
endif()
