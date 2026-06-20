#[=======================================================================[.rst:
Native Backend
--------------

Aggregator for all native (non-Rust) backend libraries.  Conditionally
includes the CUDA and HIP subdirectories based on
``NOVA_HAS_CUDA`` / ``NOVA_HAS_HIP``.  Creates the ``native`` static
library that collects all backend object libraries.

When neither CUDA nor HIP is enabled a stub C source file is generated
to satisfy CMake's requirement that static libraries contain at least
one source file.

Targets created:

``native``
  Static library that bundles all available native backends.

#]=======================================================================]

set(NATIVE_SOURCES "")

if(NOVA_HAS_CUDA)
    add_subdirectory(cuda)
endif()

if(NOVA_HAS_HIP)
    add_subdirectory(hip)
endif()

if(TARGET cuda_kernels_obj)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:cuda_kernels_obj>)
endif()

if(TARGET hip_kernels_obj)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:hip_kernels_obj>)
endif()

if(TARGET cuda_obj)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:cuda_obj>)
endif()

if(TARGET hip_obj)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:hip_obj>)
endif()

if(NOT NATIVE_SOURCES)
    set(NATIVE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/native_stub.c")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/native_stub.c"
        "/* Stub file to satisfy empty library requirement. */\n"
    )
endif()

add_library(native STATIC ${NATIVE_SOURCES})
set_target_properties(native PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME native
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    HIP_STANDARD 23
    HIP_STANDARD_REQUIRED ON
)

target_include_directories(native
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PUBLIC ${CMAKE_SOURCE_DIR}/ncore/native/cuda/memory
    PUBLIC ${CMAKE_SOURCE_DIR}/ncore/native/hip/memory
)

if(TARGET cuda_obj)
    target_link_libraries(native PUBLIC cuda_obj)
endif()

if(TARGET hip_obj)
    target_link_libraries(native PUBLIC hip_obj)
endif()
