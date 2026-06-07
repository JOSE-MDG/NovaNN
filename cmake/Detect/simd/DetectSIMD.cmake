#[=======================================================================[.rst:
.. module:: DetectSIMD
   :synopsis: Central orchestrator for CPU SIMD instruction detection.

Includes all individual CPU detection modules, derives aggregate flags,
and deduplicates the ``SIMD_FLAGS`` list.

This module is the single entry point for the SIMD detection subsystem.
It is included by ``NovaNNRuntime.cmake`` and should not be included
directly by ``CMakeLists.txt`` files.

**Detected variables:**

- ``SIMD_FLAGS`` — Compiler flags for all supported instruction sets
  (e.g. ``-msse4.2 -mavx -mavx2``).  Consumed by
  ``nova_configure_cpu_target()``.
- ``HAS_VNNI`` — Set to ``1`` if either ``HAS_AVX2_VNNI`` or
  ``HAS_AVX512_VNNI`` is detected.

**Included modules:**

- ``CheckInstructionSupport.cmake`` — ``check_simd()`` macro.
- ``DetectSSE.cmake`` — SSE4.2.
- ``DetectAVX.cmake`` — AVX, F16C, FMA3.
- ``DetectAVX2.cmake`` — AVX2, AVX2-VNNI.
- ``DetectAVX512.cmake`` — AVX-512 and sub-extensions.
- ``DetectAMX.cmake`` — AMX and sub-extensions.
#]=======================================================================]

set(SIMD_FLAGS "")

include("${CMAKE_SOURCE_DIR}/cmake/Utils/CheckInstructionSupport.cmake")

include("${CMAKE_SOURCE_DIR}/cmake/Detect/cpu/DetectSSE.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Detect/cpu/DetectAVX.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Detect/cpu/DetectAVX2.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Detect/cpu/DetectAVX512.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Detect/cpu/DetectAMX.cmake")

if(HAS_AVX2_VNNI OR HAS_AVX512_VNNI)
    set(HAS_VNNI 1)
else()
    set(HAS_VNNI 0)
endif()

list(REMOVE_DUPLICATES SIMD_FLAGS)
message(STATUS "SIMD flags: ${SIMD_FLAGS}")
