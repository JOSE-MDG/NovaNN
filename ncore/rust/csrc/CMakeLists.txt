#[=======================================================================[.rst:
Rust FFI Bridge
---------------

Build the C++ FFI bridge library for the Rust ``ncore_memory`` crate.
This static library provides the C/C++ entry points that Rust calls
into via ``extern "C"``.

Targets created:

``rustcsrc``
  Static library containing ``ffi.cpp`` and ``admin.cpp``.

#]=======================================================================]

set(RUSTCSRC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ffi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/admin.cpp
)

add_library(rustcsrc STATIC ${RUSTCSRC_SOURCES})

set_target_properties(rustcsrc PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    HIP_STANDARD 23
    HIP_STANDARD_REQUIRED ON
    OUTPUT_NAME rustcsrc
)

target_include_directories(rustcsrc
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/rust/csrc
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rustcsrc PRIVATE native)

nova_configure_build_flags(rustcsrc)

nova_configure_cpu_target(rustcsrc)

if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(rustcsrc)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(rustcsrc)
endif()
