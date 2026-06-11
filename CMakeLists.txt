#[=======================================================================[.rst:
.. module:: NovaNN
   :synopsis: Root build configuration for the NovaNN project.

Defines the **NovaNN** project (v5.0.0) and builds the final shared
library ``libnova.so`` (Linux) or ``nova.dll`` (Windows).

**Project settings:**

- C standard: C23 (required)
- C++ standard: C++23 (required)
- C++ extensions: disabled
- Position-independent code: enabled globally
- Default build type: Release

**Options:**

- ``USE_CUDA`` — Compile with NVIDIA CUDA support (default: ``OFF``).
- ``USE_HIP``  — Compile with AMD HIP/ROCm support (default: ``OFF``).
- ``USE_LTO``  — Enable Link-Time Optimization (default: ``ON``).
- ``USE_ASAN`` — Enable AddressSanitizer (default: ``OFF``).
- ``USE_UBSAN`` — Enable UndefinedBehaviorSanitizer (default: ``OFF``).

**Targets:**

- ``nova`` — Shared library assembled from ``ncore_obj`` and
  ``autograd_obj`` OBJECT libraries, linked against ``ncore_memory``
  (Rust static library) via ``WHOLE_ARCHIVE``.

**Dependencies:**

- ``ncore_obj``    — C runtime (repr + core) from ``ncore/CMakeLists.txt``.
- ``autograd_obj`` — C++ autograd engine from ``ncore/CMakeLists.txt``.
- ``ncore_memory`` — Rust memory management from ``ncore/rust/CMakeLists.txt``.
- ``dl``, ``m``     — System libraries.
#]=======================================================================]

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(NovaNN VERSION 5.0.0 LANGUAGES C CXX)

if(MSVC)
    message(FATAL_ERROR
        "MSVC is not supported. "
        "Please use GCC, G++, Clang, or Clang++ (mingw64/ucrt64)."
    )
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_CUDA "Compile with NVIDIA CUDA support" OFF)
option(USE_HIP "Compile with AMD HIP/ROCm support" OFF)
option(USE_LTO "Enable Link-Time Optimization" ON)
option(USE_ASAN "Enable AddressSanitizer" OFF)
option(USE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

include(Modules/NovaNNRuntime)

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/config.h.in"
    "${CMAKE_BINARY_DIR}/config.h"
)

add_subdirectory(ncore)

add_library(nova SHARED
    $<TARGET_OBJECTS:ncore_obj>
    $<TARGET_OBJECTS:autograd_obj>
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

nova_configure_build_flags(nova)
nova_configure_linker(nova)

nova_configure_cpu_target(nova)

if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(nova)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(nova)
endif()

set_target_properties(nova PROPERTIES
    OUTPUT_NAME "nova"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
