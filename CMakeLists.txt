#[[
    @file CMakeLists.txt
    @brief NovaNN top-level build configuration.

    Orchestrates the entire project build:
      - Sets C23/CXX23 language standards.
      - Detects CPU SIMD capabilities, threading back-ends (pthreads, OpenMP),
        and optional GPU back-ends (CUDA, HIP) via included modules.
      - Generates config.h from cmake/config.h.in with all detected feature
        macros.
      - Defines reusable COMPILE_FLAGS / DEBUG_FLAGS / RELEASE_FLAGS.
      - Builds the nova shared library by merging two object libraries
        (ncore_core_obj from C sources, ncore_autograd_obj from C++) and
        whole-archiving the Rust ncore_memory static library.

    @note The Rust static library (ncore_memory) is linked with
          `$<LINK_LIBRARY:WHOLE_ARCHIVE,ncore_memory>` to guarantee that
          all FFI symbols are visible to Rust's build.rs regardless of
          linker garbage-collection.

    Root CMakeLists.txt — all subdirectories inherit variables defined here.

    Requires : CMake 3.27+, C23, CXX23, Cargo (Rust)
    Outputs  : libnova.so
    See also : cmake/modules/NovaNNRuntime.cmake — runtime detection entry point
]]
cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(NovaNN VERSION 5.0.0 LANGUAGES C CXX)

# Standards
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Backend selection
option(USE_CUDA "Compile with NVIDIA CUDA support" ON)
option(USE_HIP "Compile with AMD HIP/ROCm support" ON)

# Paths
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

# Runtime: SIMD + threading + GPU detection
# Replaces the old bare include(detect/simd/DetectSIMD) + find_package(Threads)
include(modules/NovaNNRuntime)

# config.h
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/config.h.in"
    "${CMAKE_BINARY_DIR}/config.h"
)

# Compile flags
set(COMPILE_FLAGS
    -Wall -Wextra -Wpedantic
    -Wshadow -Wcast-align -Wconversion -Wfloat-equal
    -Wformat=2 -Wimplicit-fallthrough -Wnull-dereference
    -Wpointer-arith -Wsign-conversion -Wundef
    -Wuninitialized -Wunused
    -Wno-missing-field-initializers -Wno-unused-parameter
)
set(DEBUG_FLAGS -g -fno-omit-frame-pointer)
set(RELEASE_FLAGS -O3 -march=native -ffast-math)

# nova shared library
add_subdirectory(ncore)

add_library(nova SHARED
    $<TARGET_OBJECTS:ncore_core_obj>
    $<TARGET_OBJECTS:ncore_autograd_obj>
)

target_link_libraries(nova
    PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,ncore_memory>"
    ${CMAKE_DL_LIBS}
    m
)

target_include_directories(nova
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_BINARY_DIR}
)

target_compile_options(nova PRIVATE
    ${COMPILE_FLAGS}
    ${SIMD_FLAGS}
    $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)

novaNN_configure_cpu_target(nova)
novaNN_configure_cuda_target(nova)
novaNN_configure_hip_target(nova)

target_compile_definitions(nova PRIVATE
    NOVA_COMPILE_CUDA=$<BOOL:${USE_CUDA}>
    NOVA_COMPILE_HIP=$<BOOL:${USE_HIP}>
)

set_target_properties(nova PROPERTIES
    OUTPUT_NAME "nova"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)


