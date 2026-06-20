#[=======================================================================[.rst:
NovaNNRuntime
-------------

Top-level orchestrator module for the NovaNN build system.  Includes
all detection and configuration modules, then prints a summary of
available runtime capabilities.

This module includes:

- ``DetectSIMD`` — CPU SIMD instruction detection.
- ``NovaNNCPU`` — CPU target configuration (threading, OpenMP).
- ``NovaNNBuildFlags`` — compiler warnings, optimization, LTO,
  sanitizers.
- ``NovaNNCUDA`` — CUDA backend (when ``USE_CUDA`` is ``ON``).
- ``NovaNNHIP`` — HIP/ROCm backend (when ``USE_HIP`` is ``ON``).

When ``USE_CUDA`` or ``USE_HIP`` is ``OFF`` the module defines empty
stub functions for the corresponding ``nova_configure_*`` API so that
callers do not need to guard every call site.

This module prints a status summary to the CMake output including
SIMD flags, threading backends, GPU backends, LTO, and sanitizer
status.

#]=======================================================================]

include("${CMAKE_SOURCE_DIR}/cmake/Detect/simd/DetectSIMD.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Modules/NovaNNCPU.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Modules/NovaNNBuildFlags.cmake")

if(USE_CUDA)
    include("${CMAKE_SOURCE_DIR}/cmake/Modules/NovaNNCUDA.cmake")
else()
    set(NOVA_HAS_CUDA 0 CACHE INTERNAL "")

    function(nova_configure_cuda_runtime_target TARGET)
    endfunction()

    function(nova_configure_cuda_kernels_target TARGET)
    endfunction()

    function(nova_configure_cuda_target TARGET)
    endfunction()
endif()

if(USE_HIP)
    include("${CMAKE_SOURCE_DIR}/cmake/Modules/NovaNNHIP.cmake")
else()
    set(NOVA_HAS_HIP 0 CACHE INTERNAL "")

    function(nova_configure_hip_runtime_target TARGET)
    endfunction()

    function(nova_configure_hip_kernels_target TARGET)
    endfunction()

    function(nova_configure_hip_target TARGET)
    endfunction()
endif()

message(STATUS "NovaNN Runtime capabilities:")
message(STATUS "  CPU SIMD flags : ${SIMD_FLAGS}")

if(NOVA_HAS_PTHREADS)
    message(STATUS "  pthreads        : Enabled")
else()
    message(STATUS "  pthreads        : Disabled")
endif()

if(NOVA_HAS_OPENMP)
    message(STATUS "  OpenMP          : Enabled")
else()
    message(STATUS "  OpenMP          : Disabled")
endif()

if(USE_CUDA)
    if(NOVA_HAS_CUDA)
        message(STATUS "  CUDA            : Enabled")
    else()
        message(STATUS "  CUDA            : Disabled")
    endif()
endif()

if(USE_HIP)
    if(NOVA_HAS_HIP)
        message(STATUS "  HIP             : Enabled")
    else()
        message(STATUS "  HIP             : Disabled")
    endif()
endif()

if(NOVA_HAS_LTO)
    message(STATUS "  LTO             : Enabled")
else()
    message(STATUS "  LTO             : Disabled")
endif()

if(NOVA_HAS_ASAN)
    message(STATUS "  ASan            : Enabled")
endif()

if(NOVA_HAS_UBSAN)
    message(STATUS "  UBSan           : Enabled")
endif()
