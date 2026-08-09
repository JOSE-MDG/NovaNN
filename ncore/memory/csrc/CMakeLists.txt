#[=======================================================================[.rst:
Rust FFI Bridge
---------------

Build the C++ FFI bridge library for the Rust ``ncore_memory`` crate.
This static library provides the C/C++ entry points that Rust calls
into via ``extern "C"``.

Targets created:

``memorycsrc`` — alias ``ncore::memory::csrc``
  Static library containing ``ffi.cpp`` and ``admin.cpp``.

#]=======================================================================]

set(RUSTCSRC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ffi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/admin.cpp
)

add_library(memorycsrc STATIC ${RUSTCSRC_SOURCES})
add_library(ncore::memory::csrc ALIAS memorycsrc)

set_target_properties(memorycsrc PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME memorycsrc
)

if(NOVA_HAS_CUDA)
    set_target_properties(memorycsrc PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(memorycsrc
        PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/cuda/memory
        PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/cuda
    )
endif()

if(NOVA_HAS_HIP)
    set_target_properties(memorycsrc PROPERTIES
        HIP_STANDARD 23
        HIP_STANDARD_REQUIRED ON
    )
    target_include_directories(memorycsrc
        PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/hip/memory
        PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/hip
    )
endif()

target_include_directories(memorycsrc
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(memorycsrc PRIVATE ncore::native)

nova_configure_build_flags(memorycsrc)
nova_configure_cpu_target(memorycsrc)

if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(memorycsrc)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(memorycsrc)
endif()
