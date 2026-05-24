#[[
    @file DetectSIMD.cmake
    @brief SIMD detection orchestrator.

    Initialises SIMD_FLAGS, then runs every CPU-family detector in strict
    dependency order (SSE → AVX → AVX2 → AVX-512 → AMX).  Each detector
    uses the shared check_simd macro and appends to SIMD_FLAGS on success.

    After inclusion:
      - SIMD_FLAGS   — deduplicated list of -m flags for all detected extensions
      - HAS_SSE4_2   — SSE4.2 (CRC32)
      - HAS_AVX      — 256-bit vector registers
      - HAS_F16C     — FP16 conversion (requires AVX)
      - HAS_FMA3     — FMA3 instructions (requires AVX)
      - HAS_AVX2     — 256-bit integer SIMD
      - HAS_AVX2_VNNI / HAS_AVX2_INT8 — AVX2 VNNI (requires AVX2)
      - HAS_AVX512F + BW, DQ, VL, VNNI, FP16, BF16 — AVX-512 extensions
      - HAS_AMX + AMX_FP16, AMX_BF16, AMX_INT8 — AMX extensions (requires AMX-TILE)
      - HAS_VNNI     — aggregate: AVX2-VNNI or AVX512-VNNI

    Includes:
      1. CheckInstructionSupport.cmake — defines the check_simd macro
      2. DetectSSE.cmake
      3. DetectAVX.cmake
      4. DetectAVX2.cmake
      5. DetectAVX512.cmake
      6. DetectAMX.cmake

    Usage:
        include(detect/simd/DetectSIMD)

    See also : CheckInstructionSupport.cmake — check_simd macro definition
               NovaNNCPU.cmake — consumes SIMD_FLAGS for target configuration
]]

set(SIMD_FLAGS "")

include("${CMAKE_CURRENT_LIST_DIR}/../../utils/CheckInstructionSupport.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/../cpu/DetectSSE.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../cpu/DetectAVX.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../cpu/DetectAVX2.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../cpu/DetectAVX512.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../cpu/DetectAMX.cmake")

#[[
    @var HAS_VNNI
    @brief Aggregate: any VNNI support (AVX2-VNNI or AVX512-VNNI)
]]
if(HAS_AVX2_VNNI OR HAS_AVX512_VNNI)
    set(HAS_VNNI 1)
else()
    set(HAS_VNNI 0)
endif()

list(REMOVE_DUPLICATES SIMD_FLAGS)
message(STATUS "SIMD flags: ${SIMD_FLAGS}")
