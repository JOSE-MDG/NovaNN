#[=======================================================================[.rst:
.. module:: ncore
   :synopsis: Core C and C++ object libraries for NovaNN.

Defines the two OBJECT libraries that are assembled into the final
``nova`` shared library.

**Targets:**

- ``ncore_obj``   — OBJECT library from all ``.c`` files in
  ``ncore/src/repr/`` and ``ncore/src/core/``.
- ``autograd_obj`` — OBJECT library from all ``.cpp`` files in
  ``ncore/src/autograd/``.

Both targets have ``POSITION_INDEPENDENT_CODE ON`` and receive the
same compile flags, SIMD flags, and backend configuration as the
root ``nova`` target.

**Subdirectories:**

- ``rust/`` — Builds the Rust memory subsystem.
#]=======================================================================]

add_subdirectory(rust)

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/ncore/src/repr/*.c"
    "${CMAKE_SOURCE_DIR}/ncore/src/core/*.c"
)

add_library(ncore_obj OBJECT ${CORE_SOURCES})

set_target_properties(ncore_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ncore_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/src>
    PRIVATE ${CMAKE_BINARY_DIR}
)

target_compile_options(ncore_obj PRIVATE
    ${COMPILE_FLAGS}
    ${SIMD_FLAGS}
    $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)

nova_configure_cpu_target(ncore_obj)

if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(ncore_obj)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(ncore_obj)
endif()

# Autograd C++ implementation
file(GLOB_RECURSE AUTOGRAD_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/ncore/src/autograd/*.cpp"
)

add_library(autograd_obj OBJECT ${AUTOGRAD_SOURCES})

set_target_properties(autograd_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(autograd_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_BINARY_DIR}
)

target_compile_options(autograd_obj PRIVATE
    ${COMPILE_FLAGS}
    ${SIMD_FLAGS}
    $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)

nova_configure_cpu_target(autograd_obj)

if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(autograd_obj)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(autograd_obj)
endif()
