#[[
    @file NovaNNRuntime.cmake
    @brief Runtime-detection orchestrator — single entry point for all
           feature-detection modules.

    After inclusion, the following are available globally:
      - SIMD_FLAGS               — list of -m flags for enabled SIMD extensions
      - HAS_*                    — per-feature capability flags (HAS_AVX2, HAS_AMX, ...)
      - NOVA_HAS_PTHREADS        — 1 if POSIX threads are available
      - NOVA_HAS_OPENMP          — 1 if OpenMP C/CXX are available
      - NOVA_HAS_CUDA            — 1 if CUDA toolkit ≥12.3 is found
      - NOVA_HAS_HIP             — 1 if ROCm ≥6.2 is found
      - novaNN_configure_cpu_target()   — apply SIMD + threading to a target
      - novaNN_configure_cuda_target()  — apply CUDA to a target (no-op if no CUDA)
      - novaNN_configure_hip_target()   — apply HIP to a target (no-op if no HIP)

    Include order:
      1. DetectSIMD.cmake        — CPU SIMD instruction-set detection
      2. NovaNNCPU.cmake         — SIMD + threading target configuration function
      3. NovaNNCUDA.cmake        — CUDA detection + target configuration function
      4. NovaNNHIP.cmake         — HIP detection + target configuration function

    Usage:
        include(modules/NovaNNRuntime)

    See also : config.h.in — feature macros consumed by C source code
]]
include("${CMAKE_SOURCE_DIR}/cmake/detect/simd/DetectSIMD.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/modules/NovaNNCPU.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/modules/NovaNNCUDA.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/modules/NovaNNHIP.cmake")

message(STATUS "NovaNN Runtime capabilities:")
message(STATUS "  CPU SIMD flags : ${SIMD_FLAGS}")
message(STATUS "  pthreads       : ${NOVA_HAS_PTHREADS}")
message(STATUS "  OpenMP         : ${NOVA_HAS_OPENMP}")
message(STATUS "  CUDA           : ${NOVA_HAS_CUDA}")
message(STATUS "  HIP            : ${NOVA_HAS_HIP}")
