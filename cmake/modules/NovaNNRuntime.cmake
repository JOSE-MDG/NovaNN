#[[
    @file NovaNNRuntime.cmake
    @brief Runtime-detection orchestrator — single entry point for all
           feature-detection modules.

    After inclusion, the following are available globally:
      - SIMD_FLAGS               — list of -m flags for enabled SIMD extensions
      - HAS_*                    — per-feature capability flags (HAS_AVX2, HAS_AMX, ...)
      - NOVA_HAS_PTHREADS        — 1 if POSIX threads are available
      - NOVA_HAS_OPENMP          — 1 if OpenMP C/CXX are available
      - NOVA_HAS_CUDA            — 1 if CUDA toolkit ≥12.6 is found
      - NOVA_HAS_HIP             — 1 if ROCm ≥6.2 is found
      - novaNN_configure_cpu_target()   — apply SIMD + threading to a target
      - novaNN_configure_cuda_target()  — apply CUDA to a target (no-op if no CUDA)
      - novaNN_configure_hip_target()   — apply HIP to a target (no-op if no HIP)

    The USE_CUDA / USE_HIP options (set in the root CMakeLists.txt) gate the
    optional GPU back-ends. CPU is always compiled as the mandatory baseline.

    Include order:
      1. DetectSIMD.cmake        — CPU SIMD instruction-set detection
      2. NovaNNCPU.cmake         — SIMD + threading target configuration function
      3. NovaNNCUDA.cmake        — CUDA detection + target configuration function
      4. NovaNNHIP.cmake         — HIP detection + target configuration function

    Usage:
        include(modules/NovaNNRuntime)

    See also : config.h.in — feature macros consumed by C source code
]]

# ── CPU (always enabled — mandatory baseline) ─────────────
include("${CMAKE_SOURCE_DIR}/cmake/detect/simd/DetectSIMD.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/modules/NovaNNCPU.cmake")

# ── CUDA ───────────────────────────────────────────────────
if(USE_CUDA)
    include("${CMAKE_SOURCE_DIR}/cmake/modules/NovaNNCUDA.cmake")
else()
    set(NOVA_HAS_CUDA 0)
    function(novaNN_configure_cuda_target TARGET)
        # No-op — CUDA support disabled by user
    endfunction()
endif()

# ── HIP ────────────────────────────────────────────────────
if(USE_HIP)
    include("${CMAKE_SOURCE_DIR}/cmake/modules/NovaNNHIP.cmake")
else()
    set(NOVA_HAS_HIP 0)
    function(novaNN_configure_hip_target TARGET)
        # No-op — HIP support disabled by user
    endfunction()
endif()

message(STATUS "NovaNN Runtime capabilities:")
message(STATUS "  CPU SIMD flags : ${SIMD_FLAGS}")
message(STATUS "  pthreads       : ${NOVA_HAS_PTHREADS}")
message(STATUS "  OpenMP         : ${NOVA_HAS_OPENMP}")
if(USE_CUDA)
    message(STATUS "  CUDA           : ${NOVA_HAS_CUDA}")
else()
    message(STATUS "  CUDA           : Disabled by user")
endif()
if(USE_HIP)
    message(STATUS "  HIP            : ${NOVA_HAS_HIP}")
else()
    message(STATUS "  HIP            : Disabled by user")
endif()
