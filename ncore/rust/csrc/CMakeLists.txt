#[[
    @file ncore/rust/csrc/CMakeLists.txt
    @brief C++ FFI + device back-end static library (librustcsrc.a).

    Always compiled:
      - ffi.cpp          — extern "C" bridge: reserve / release / memcpy wrappers
      - device/admin.cpp — back-end admin: get_device_backend(), device_reserve(),
                           device_release(), device_memcpy()

    Conditionally included (guarded by NOVA_HAS_CUDA / NOVA_HAS_HIP):
      - device/cuda/     — cuda_allocator.cpp, cuda_io.cpp
      - device/hip/      — hip_allocator.cpp, hip_io.cpp

    The resulting librustcsrc.a is consumed by build.rs via the RUSTCSRC_DIR
    and RUSTCSRC_NAME environment variables set in the parent CMakeLists.txt.

    Compile flags (COMPILE_FLAGS, SIMD_FLAGS) and target configurations
    (novaNN_configure_cpu/cuda/hip_target) are inherited from the root scope.

    Requires : NOVA_HAS_CUDA, NOVA_HAS_HIP (set by NovaNNRuntime.cmake)
    Outputs  : librustcsrc.a (static library target: rustcsrc)
    See also : ncore/rust/CMakeLists.txt — Cargo integration with rustcsrc
]]

set(RUSTCSRC_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/device/admin.cpp"
)

if(NOVA_HAS_CUDA)
    list(APPEND RUSTCSRC_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/device/cuda/cuda_allocator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/device/cuda/cuda_io.cpp"
    )
endif()

if(NOVA_HAS_HIP)
    list(APPEND RUSTCSRC_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/device/hip/hip_allocator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/device/hip/hip_io.cpp"
    )
endif()

add_library(rustcsrc STATIC ${RUSTCSRC_SOURCES})

set_target_properties(rustcsrc PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME rustcsrc # produces librustcsrc.a
)

target_include_directories(rustcsrc
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_SOURCE_DIR}/ncore/include"
    "${CMAKE_BINARY_DIR}"
)

target_compile_options(rustcsrc PRIVATE
    ${COMPILE_FLAGS}
    ${SIMD_FLAGS}
    $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)

novaNN_configure_cpu_target(rustcsrc)
novaNN_configure_cuda_target(rustcsrc)
novaNN_configure_hip_target(rustcsrc)
