cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

#[=======================================================================[.rst:
NovaNN Project
--------------

Root CMakeLists.txt for the NovaNN, High Performance Deep Learning Framework.
Configures the project-wide build settings, detects available hardware
backends, and creates the main ``nova`` shared library.

Project Options
^^^^^^^^^^^^^^^

``USE_CUDA``
  Enable NVIDIA CUDA support.  Default: ``OFF``.

``USE_HIP``
  Enable AMD HIP/ROCm support.  Default: ``OFF``.

``USE_LTO``
  Enable Link-Time Optimization.  Default: ``ON``.

``USE_ASAN``
  Enable AddressSanitizer.  Default: ``OFF``.

``USE_UBSAN``
  Enable UndefinedBehaviorSanitizer.  Default: ``OFF``.

Build Targets
^^^^^^^^^^^^^

``nova``
  The main shared library.  Aggregates ``ncore_obj``, ``autograd_obj``,
  and ``ncore_memory`` (Rust).  Links against ``dl`` and ``m``.

Subdirectories
^^^^^^^^^^^^^^

- ``ncore/`` — Core module containing native backends, Rust crate, and
object libraries.

#]=======================================================================]
project(NovaNN VERSION 5.0.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(Utils/CheckCompilerVersion)
include(Modules/NovaNNRuntime)

if(MSVC)
    message(FATAL_ERROR
        "MSVC is not supported. "
        "Please use GCC, G++, Clang, or Clang++ (MINGW64/UCRT64)."
    )
endif()

check_min_compiler_version()

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

if(USE_CUDA)
    enable_language(CUDA)
endif()

if(USE_HIP)
    enable_language(HIP)
endif()

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
    -Wl,--whole-archive
    ncore_memory
    native
    -Wl,--no-whole-archive
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
