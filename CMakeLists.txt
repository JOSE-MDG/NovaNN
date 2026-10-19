cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

#[=======================================================================[.rst:
NovaNN Project
--------------

Root CMakeLists.txt.  Configures the project-wide build settings,
detects hardware backends, and creates the main ``nova`` shared library.

Project Options
^^^^^^^^^^^^^^^

``USE_CUDA``
  Enable NVIDIA CUDA support.  Default: ``OFF``.  Cannot be combined
  with ``USE_HIP``.  Also enables the ``cuda`` vcpkg manifest feature,
  so the ``cuda`` helper port is installed and verified during the
  vcpkg install step.

``USE_HIP``
  Enable AMD HIP/ROCm support.  Default: ``OFF``.  Cannot be combined
  with ``USE_CUDA``.  Not available on Windows.  Also enables the
  ``rocm`` vcpkg manifest feature.

``USE_LTO``
  Enable Link-Time Optimization.  Default: ``ON``.

``USE_ASAN``
  Enable AddressSanitizer.  Default: ``OFF``.

``USE_UBSAN``
  Enable UndefinedBehaviorSanitizer.  Default: ``OFF``.

Platform and Compiler Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Linux:** GCC and Clang
- **Windows:** Clang/LLVM (clang-cl)

Invalid combinations are rejected at configure time with a fatal error.
#]=======================================================================]
option(USE_CUDA "Compile with NVIDIA CUDA support" OFF)
option(USE_HIP "Compile with AMD HIP/ROCm support" OFF)

# The vcpkg toolchain file loads during project() and reads
# VCPKG_MANIFEST_FEATURES at that point, so the mapping has to happen
# before it. The helper ports install and verify the backend SDK.
if(USE_CUDA)
    list(APPEND VCPKG_MANIFEST_FEATURES cuda)
endif()
if(USE_HIP)
    list(APPEND VCPKG_MANIFEST_FEATURES rocm)
endif()

project(NovaNN VERSION 5.0.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Fail fast when the build keeps re-running CMake: the guard counts the
# configures of this build tree and aborts with a diagnostic.
include(Modules/NovaNNReconfigureGuard)
nova_guard_reconfigure_loop()

option(USE_LTO "Enable Link-Time Optimization" ON)
option(USE_ASAN "Enable AddressSanitizer" OFF)
option(USE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# Reject invalid option combinations before any backend is enabled.
if(USE_CUDA AND USE_HIP)
    message(FATAL_ERROR
        "CUDA and HIP cannot be enabled simultaneously. "
        "Set at most one of USE_CUDA or USE_HIP to ON."
    )
endif()

if(USE_HIP AND WIN32)
    message(FATAL_ERROR
        "HIP is not supported on Windows. "
        "Set USE_HIP to OFF or use a Linux host."
    )
endif()

if(USE_CUDA)
    if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
        set(CMAKE_CUDA_ARCHITECTURES
            "75;80;86;89;90;100;103;110;120;121"
            CACHE STRING "CUDA target architectures for NovaNN")
    endif()
    enable_language(CUDA)
endif()


if(USE_HIP)
    if(NOT DEFINED CACHE{CMAKE_HIP_ARCHITECTURES})
        set(CMAKE_HIP_ARCHITECTURES
            "gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
            CACHE STRING "HIP target architectures for NovaNN")
    endif()
    enable_language(HIP)
endif()

include(GNUInstallDirs)
include(Utils/CheckCompilerVersion)
include(Modules/NovaNNRuntime)

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

# clangd searches parent directories for compile_commands.json, so
# symlink the build tree's copy into the source root.  Skipped on
# Windows, where creating symlinks requires elevated privileges.
if(NOT WIN32)
    set(NOVA_CLANGD_DB "${CMAKE_SOURCE_DIR}/compile_commands.json")
    if(IS_SYMLINK "${NOVA_CLANGD_DB}")
        file(REMOVE "${NOVA_CLANGD_DB}")
    endif()
    if(NOT EXISTS "${NOVA_CLANGD_DB}")
        file(CREATE_LINK
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${NOVA_CLANGD_DB}"
            SYMBOLIC
        )
    endif()
endif()

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/config.h.in"
    "${CMAKE_BINARY_DIR}/config.h"
)

add_subdirectory(ncore)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(ncore/tests)
endif()

add_library(nova SHARED
    $<TARGET_OBJECTS:ncore::obj::core>
    $<TARGET_OBJECTS:ncore::obj::autograd>
    $<TARGET_OBJECTS:ncore::obj::dtypes>
)

if(WIN32)
    set_target_properties(nova PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)

    set(NOVA_DEF "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nova.dir/nova.exports.def")
    set(NOVA_SCANLIST "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nova.dir/nova.scanlist.txt")
    file(GENERATE OUTPUT "${NOVA_SCANLIST}" CONTENT
        "$<TARGET_OBJECTS:ncore::obj::core>$<SEMICOLON>$<TARGET_OBJECTS:ncore::obj::autograd>$<SEMICOLON>$<TARGET_OBJECTS:ncore::obj::dtypes>$<SEMICOLON>$<TARGET_FILE:ncore::memory>$<SEMICOLON>$<TARGET_FILE:ncore::native>"
    )
    add_custom_command(TARGET nova PRE_LINK
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_LIST="${NOVA_SCANLIST}"
            -DOUTPUT_DEF="${NOVA_DEF}"
            -DCMAKE_NM="${CMAKE_NM}"
            -P "${CMAKE_SOURCE_DIR}/cmake/GenExportDef.cmake"
        COMMENT "Generating exports.def from objects + static libs"
    )
    target_link_options(nova PRIVATE /DEF:${NOVA_DEF})
endif()

# `--whole-archive` / `/WHOLEARCHIVE` ensure every object in the static libs
# is pulled into the shared library, not just those with referenced symbols.
if(WIN32)
    # lld-link uses /WHOLEARCHIVE:<lib>
    target_link_options(nova PRIVATE
        /WHOLEARCHIVE:$<TARGET_FILE:ncore::memory>
        /WHOLEARCHIVE:$<TARGET_FILE:ncore::native>
    )
    target_link_libraries(nova PRIVATE ncore::memory ncore::native)
else()
    target_link_libraries(nova PRIVATE
        -Wl,--whole-archive
        ncore::memory
        ncore::native
        -Wl,--no-whole-archive
    )
endif()

if(NOT WIN32)
    target_link_libraries(nova PRIVATE ${CMAKE_DL_LIBS} m)
else()
    target_link_libraries(nova PRIVATE
        ws2_32
        ntdll
        userenv
        bcrypt
    )
endif()

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
    LINKER_LANGUAGE CXX
)
