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
this module defines the ``nova_codegen`` custom target (aliased as
``nova::codegen``) that runs the auto-code generation command
``uv run tools/codegen/generate.py gen --all`` at build time.

The list of generated files consumed as ``OUTPUT`` by the codegen
custom command (``NOVA_CODEGEN_FILES``).
``uv run tools/codegen/generate.py gen --all --list-outputs``, which
prints the candidate paths one per line without generating anything.

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
elseif(NOVA_HAS_WIN32_THREADS)
    message(STATUS "  Win32 threads   : Enabled")
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

execute_process(
    COMMAND uv run -q tools/codegen/generate.py gen --all --list-outputs
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE NOVA_CODEGEN_RC
    OUTPUT_VARIABLE NOVA_CODEGEN_RAW
    ERROR_VARIABLE NOVA_CODEGEN_ERR
)

if(NOT NOVA_CODEGEN_RC EQUAL 0)
    message(FATAL_ERROR
        "Codegen output discovery failed (exit code ${NOVA_CODEGEN_RC}):\n"
        "${NOVA_CODEGEN_ERR}"
        "Command: uv run -q tools/codegen/generate.py gen --all "
        "--list-outputs\n"
        "Ensure 'uv' is installed and the Python environment is synced "
        "(run 'uv sync' at the project root), then re-run CMake configure."
    )
endif()

set(NOVA_CODEGEN_FILES)
string(REGEX REPLACE "[\r\n]+" ";" NOVA_CODEGEN_LIST "${NOVA_CODEGEN_RAW}")
foreach(_nova_codegen_path IN LISTS NOVA_CODEGEN_LIST)
    if(_nova_codegen_path)
        list(APPEND NOVA_CODEGEN_FILES
            "${CMAKE_SOURCE_DIR}/${_nova_codegen_path}"
        )
    endif()
endforeach()

if(NOT NOVA_CODEGEN_FILES)
    message(FATAL_ERROR
        "Codegen output discovery returned no output files.\n"
        "Check that tools/codegen/scripts/ contains gen_*.py files that "
        "call register_engine() at import time, then run manually:\n"
        "  uv run tools/codegen/generate.py gen --all --list-outputs"
    )
endif()

file(GLOB_RECURSE NOVA_CODEGEN_DEPS CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/tools/codegen/*.py"
    "${CMAKE_SOURCE_DIR}/tools/codegen/templates/*.jinja"
)

add_custom_command(
    OUTPUT ${NOVA_CODEGEN_FILES}
    COMMAND uv run tools/codegen/generate.py gen --all --keep-going --run-formatters
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    DEPENDS ${NOVA_CODEGEN_DEPS}
    COMMENT "Codegen: generating auto-generated code using uv toolchain ..."
    VERBATIM
)

# Build target that runs the code generator at build time.
add_custom_target(nova_codegen DEPENDS ${NOVA_CODEGEN_FILES})

add_library(nova_codegen_iface INTERFACE)
add_dependencies(nova_codegen_iface nova_codegen)
add_library(nova::codegen ALIAS nova_codegen_iface)
