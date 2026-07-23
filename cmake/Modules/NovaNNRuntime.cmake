#[=======================================================================[.rst:
NovaNNRuntime
-------------

Top-level orchestrator module for the NovaNN build system.  Includes
all detection and configuration modules, then prints a summary of
available runtime capabilities.

This module includes:

- ``DetectSIMD`` — CPU SIMD instruction detection.
- ``NovaNNCPU`` — CPU target configuration (threading, OpenMP).
- ``NovaNNBuildFlags`` — compiler warnings, optimization, LTO, sanitizers.
- ``DetectGTest`` — GoogleTest testing framework detection (only when
  ``BUILD_TESTING`` is ``ON``).
- ``NovaNNCUDA`` — CUDA backend (when ``USE_CUDA`` is ``ON``).
- ``NovaNNHIP`` — HIP/ROCm backend (when ``USE_HIP`` is ``ON``).

When ``USE_CUDA`` or ``USE_HIP`` is ``OFF`` the module defines empty
stub functions for the corresponding ``nova_configure_*`` API so that
callers do not need to guard every call site.  The same is done for
GTest when ``BUILD_TESTING`` is ``OFF`` or the library is not found.

After printing a status summary to the CMake output (SIMD flags,
threading backends, GPU backends, GTest, LTO, and sanitizer status),
this module runs the auto-code generation command
``uv run tools/codegen/generate.py gen -a`` at configure time.

#]=======================================================================]

include(Detect/simd/DetectSIMD)
include(Modules/NovaNNCPU)
include(Modules/NovaNNBuildFlags)

if(BUILD_TESTING)
    include(Detect/testing/DetectGTest)

    if(NOT NOVA_HAS_GTEST)
        set(NOVA_HAS_GTEST 0 CACHE INTERNAL "")

        function(nova_configure_gtest_target TARGET)
        endfunction()
    endif()
else()
    set(NOVA_HAS_GTEST 0 CACHE INTERNAL "")

    function(nova_configure_gtest_target TARGET)
    endfunction()
endif()

if(USE_CUDA)
    include(Modules/NovaNNCUDA)
else()
    set(NOVA_HAS_CUDA 0 CACHE INTERNAL "")

    function(nova_configure_cuda_runtime_target TARGET)
    endfunction()

    function(nova_configure_cuda_includes_target TARGET)
    endfunction()

    function(nova_configure_cuda_kernels_target TARGET)
    endfunction()

    function(nova_configure_cuda_target TARGET)
    endfunction()
endif()

if(USE_HIP)
    include(Modules/NovaNNHIP)
else()
    set(NOVA_HAS_HIP 0 CACHE INTERNAL "")

    function(nova_configure_hip_runtime_target TARGET)
    endfunction()

    function(nova_configure_hip_includes_target TARGET)
    endfunction()

    function(nova_configure_hip_kernels_target TARGET)
    endfunction()

    function(nova_configure_hip_target TARGET)
    endfunction()
endif()

message(STATUS "NovaNN Runtime capabilities:")
message(STATUS "  CPU SIMD flags  : ${SIMD_FLAGS}")

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

if(BUILD_TESTING)
    if(NOVA_HAS_GTEST)
        message(STATUS "  GTest           : Enabled")
    else()
        message(STATUS "  GTest           : Disabled")
    endif()
endif()

# Code generation step via uv toolchain
message(STATUS "Codegen: generating auto-generated code using uv toolchain ...")

execute_process(
    COMMAND uv run tools/codegen/generate.py gen --all --keep-going --run-formatters
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _nova_codegen_result
    OUTPUT_VARIABLE _nova_codegen_output
    ERROR_VARIABLE _nova_codegen_error
)

if(_nova_codegen_result EQUAL 0)
    message(STATUS "Codegen: auto-generated code generated successfully ✓")
else()
    message(WARNING
        "Codegen: failed with exit code ${_nova_codegen_result}"
    )

    if(_nova_codegen_output)
        message(STATUS "Codegen output: ${_nova_codegen_output}")
    endif()

    if(_nova_codegen_error)
        message(STATUS "Codegen error: ${_nova_codegen_error}")
    endif()
endif()

unset(_nova_codegen_result)
unset(_nova_codegen_output)
unset(_nova_codegen_error)
