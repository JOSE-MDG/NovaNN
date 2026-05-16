#[[
    @file ncore/CMakeLists.txt
    @brief NovaNN core library build configuration.
    @details Builds ncore_core_obj from C source files, applies compile
             flags and standards inherited from parent.

    Requires: Parent CMakeLists.txt defines COMPILE_FLAGS
    Output: ncore_core_obj (object library)
]]
add_subdirectory(rust)

file(GLOB_RECURSE SOURCES
    "${CMAKE_SOURCE_DIR}/ncore/src/core/*.c"
    "${CMAKE_SOURCE_DIR}/ncore/src/core/tables/*.c"
    "${CMAKE_SOURCE_DIR}/ncore/src/repr/*.c"
)

add_library(ncore_core_obj OBJECT
    ${SOURCES}
)

set_target_properties(ncore_core_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ncore_core_obj
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/src>
)

target_compile_options(ncore_core_obj PRIVATE
    ${COMPILE_FLAGS}
    ${SIMD_FLAGS}
    $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)
