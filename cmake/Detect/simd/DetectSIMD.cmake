#[=======================================================================[.rst:
DetectSIMD
----------

Orchestrator module that detects all supported SIMD instruction sets.
Includes the utility macro ``CheckInstructionSupport`` and every
CPU detection module (SSE, AVX, AVX2, AVX-512, AMX).  Aggregates
detected compiler flags into the ``SIMD_FLAGS`` list variable.

This module sets the following variables:

``SIMD_FLAGS``
  Aggregated compiler flags for all detected SIMD instruction sets.
  Duplicate flags are removed before the variable is finalized.

``HAS_VNNI``
  ``1`` if either ``HAS_AVX2_VNNI`` or ``HAS_AVX512_VNNI`` was
  detected, ``0`` otherwise.

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
