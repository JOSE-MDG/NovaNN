#[[
    @file ncore/CMakeLists.txt
    @brief NovaNN core-library object builds.

    Builds two object libraries consumed by the root CMakeLists.txt:

      ncore_core_obj       — All C sources under ncore/src/core/ (tables/,
                             detect/, *.c) and ncore/src/repr/ (formatting,
                             traversal, layout).  Compiled as C23.
      ncore_autograd_obj   — All C++ sources under ncore/src/autograd/.
                             Compiled as CXX23.

    Both targets inherit COMPILE_FLAGS, SIMD_FLAGS, DEBUG_FLAGS, and
    RELEASE_FLAGS from the parent scope and have novaNN_configure_cpu_target()
    applied for SIMD/pthreads/OpenMP.

    @note GLOB_RECURSE patterns use `*` which matches path separators,
          so ncore/src/repr/*.c finds .c files in all subdirectories.

    Depends on: ncore/rust/ being configured first (add_subdirectory(rust))
    Outputs   : ncore_core_obj, ncore_autograd_obj (object libraries)
    See also  : CMakeLists.txt (root) — linker that merges these objects
]]
add_subdirectory(rust)

file(GLOB_RECURSE CORE_SOURCES
    "${CMAKE_SOURCE_DIR}/ncore/src/core/tables/*.c"
    "${CMAKE_SOURCE_DIR}/ncore/src/core/detect/*.c"
    "${CMAKE_SOURCE_DIR}/ncore/src/repr/*.c"
    "${CMAKE_SOURCE_DIR}/ncore/src/core/*.c"
)

list(REMOVE_ITEM CORE_SOURCES "${CMAKE_SOURCE_DIR}/ncore/src/core/main.c")

add_library(ncore_core_obj OBJECT ${CORE_SOURCES})

set_target_properties(ncore_core_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ncore_core_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/src>
    ${CMAKE_BINARY_DIR}
)

target_compile_options(ncore_core_obj PRIVATE
    ${COMPILE_FLAGS}
    ${SIMD_FLAGS}
    $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)

target_compile_definitions(ncore_core_obj PRIVATE
    NOVA_COMPILE_CUDA=$<BOOL:${USE_CUDA}>
    NOVA_COMPILE_HIP=$<BOOL:${USE_HIP}>
)

novaNN_configure_cpu_target(ncore_core_obj)

# Autograd C++ implementation
file(GLOB_RECURSE AUTOGRAD_SOURCES
    "${CMAKE_SOURCE_DIR}/ncore/src/autograd/*.cpp"
)

add_library(ncore_autograd_obj OBJECT ${AUTOGRAD_SOURCES})

set_target_properties(ncore_autograd_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ncore_autograd_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/src>
    ${CMAKE_BINARY_DIR}
)

target_compile_options(ncore_autograd_obj PRIVATE
    ${COMPILE_FLAGS}
    ${SIMD_FLAGS}
    $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)

target_compile_definitions(ncore_autograd_obj PRIVATE
    NOVA_COMPILE_CUDA=$<BOOL:${USE_CUDA}>
    NOVA_COMPILE_HIP=$<BOOL:${USE_HIP}>
)

novaNN_configure_cpu_target(ncore_autograd_obj)
