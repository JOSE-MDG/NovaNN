#[=======================================================================[.rst:
DetectSIMD
----------

Orchestrator module that detects all supported SIMD instruction sets.
Includes the utility macro ``CheckInstructionSupport`` and every
CPU detection module (SSE, AVX, AVX2, AVX-512, AVX10.1, AVX10.2, AMX).
Aggregates detected compiler flags into the ``SIMD_FLAGS`` list
variable.

This module sets the following variables:

``SIMD_FLAGS``
  Aggregated compiler flags for all detected SIMD instruction sets.
  Duplicate flags are removed before the variable is finalized.
  Always empty under MSVC (see ``NovaNNCPU.cmake``).

``HAS_VNNI``
  ``1`` if any of ``HAS_AVX2_VNNI``, ``HAS_AVX512_VNNI``, or
  ``HAS_AVX10_1`` was detected, ``0`` otherwise.

#]=======================================================================]

set(SIMD_FLAGS "")

include(Utils/CheckInstructionSupport)

include(Detect/cpu/DetectSSE)
include(Detect/cpu/DetectAVX)
include(Detect/cpu/DetectAVX2)
include(Detect/cpu/DetectAVX512)
include(Detect/cpu/DetectAVX10.1)
include(Detect/cpu/DetectAVX10.2)
include(Detect/cpu/DetectAMX)

if(HAS_AVX2_VNNI OR HAS_AVX512_VNNI OR HAS_AVX10_1)
  set(HAS_VNNI 1)
else()
  set(HAS_VNNI 0)
endif()

list(REMOVE_DUPLICATES SIMD_FLAGS)
