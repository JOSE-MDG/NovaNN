#[=======================================================================[.rst:
.. module:: NovaNNRuntime
   :synopsis: Top-level entry point for the NovaNN build system.

This is the master orchestrator that bootstraps all runtime capability
detection and provides the ``nova_configure_*_target()`` functions
consumed by ``CMakeLists.txt`` files.

**Included subsystems:**

- ``DetectSIMD.cmake`` — SIMD instruction detection (SSE4.2, AVX,
  AVX2, AVX-512, AMX).
- ``NovaNNCPU.cmake`` — Threading detection (pthreads, OpenMP) and
  ``nova_configure_cpu_target()``.
- ``NovaNNBuildFlags.cmake`` — Compiler warning flags, C++-specific
  flags, LTO detection, sanitizer configuration, and
  ``nova_configure_build_flags()`` / ``nova_configure_linker()``.
- ``NovaNNCUDA.cmake`` — CUDA detection and
  ``nova_configure_cuda_target()`` (only if ``USE_CUDA`` is ``ON``).
- ``NovaNNHIP.cmake`` — HIP/ROCm detection and
  ``nova_configure_hip_target()`` (only if ``USE_HIP`` is ``ON``).

**Provided functions:**

- ``nova_configure_cpu_target(TARGET)`` — always available.
- ``nova_configure_build_flags(TARGET)`` — always available.
- ``nova_configure_linker(TARGET)`` — always available.
- ``nova_configure_cuda_target(TARGET)`` — no-op if CUDA is disabled.
- ``nova_configure_hip_target(TARGET)`` — no-op if HIP is disabled.

**Output:**

Prints a capability summary at configure time showing detected SIMD
flags, threading backend, LTO status, sanitizer status, and GPU
backend status.

.. code-block:: cmake

   include(Modules/NovaNNRuntime)
   nova_configure_cpu_target(my_target)
   nova_configure_build_flags(my_target)
   nova_configure_linker(my_target)
   nova_configure_cuda_target(my_target)  # no-op if CUDA disabled
#]=======================================================================]

include("${CMAKE_SOURCE_DIR}/cmake/Detect/simd/DetectSIMD.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Modules/NovaNNCPU.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Modules/NovaNNBuildFlags.cmake")

if(USE_CUDA)
    include("${CMAKE_SOURCE_DIR}/cmake/Modules/NovaNNCUDA.cmake")
else()
    set(NOVA_HAS_CUDA 0 CACHE INTERNAL "")
    function(nova_configure_cuda_target TARGET)
    endfunction()
endif()

if(USE_HIP)
    include("${CMAKE_SOURCE_DIR}/cmake/Modules/NovaNNHIP.cmake")
else()
    set(NOVA_HAS_HIP 0 CACHE INTERNAL "")
    function(nova_configure_hip_target TARGET)
    endfunction()
endif()

message(STATUS "NovaNN Runtime capabilities:")
message(STATUS "  CPU SIMD flags : ${SIMD_FLAGS}")

# PThreads
if(NOVA_HAS_PTHREADS)
    message(STATUS "  pthreads        : Enabled")
else()
    message(STATUS "  pthreads        : Disabled")
endif()

# OpenMP
if(NOVA_HAS_OPENMP)
    message(STATUS "  OpenMP          : Enabled")
else()
    message(STATUS "  OpenMP          : Disabled")
endif()

# CUDA
if(USE_CUDA)
    if(NOVA_HAS_CUDA)
        message(STATUS "  CUDA            : Enabled")
    else()
        message(STATUS "  CUDA            : Disabled")
    endif()
endif()

# HIP
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
