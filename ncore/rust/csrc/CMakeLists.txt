#[=======================================================================[.rst:
.. module:: csrc
   :synopsis: C++ FFI bridge between Rust memory subsystem and NovaNN.

Defines the ``rustcsrc`` static library containing the C++ side of the
FFI layer.  Conditionally compiles CUDA or HIP device sources based on
backend availability.

**Targets:**

- ``rustcsrc`` — STATIC library with ``POSITION_INDEPENDENT_CODE ON``
  and C++23 required.

**Sources:**

Always compiled:

- ``ffi.cpp``             — Top-level FFI dispatch.
- ``device/admin.cpp``    — Device administration.

Conditionally compiled:

- ``device/cuda/cuda_allocator.cpp`` — CUDA memory allocation.
- ``device/cuda/cuda_io.cpp``        — CUDA memory transfer.
  (Only if ``NOVA_HAS_CUDA`` is ``1``.)
- ``device/hip/hip_allocator.cpp``   — HIP memory allocation.
- ``device/hip/hip_io.cpp``          — HIP memory transfer.
  (Only if ``NOVA_HAS_HIP`` is ``1``.)

**Include directories:**

- ``PUBLIC:`` ``ncore/include`` (project-wide headers).
- ``PRIVATE:`` ``ncore/rust/csrc`` and current source directory.
#]=======================================================================]

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
    OUTPUT_NAME rustcsrc
)

target_include_directories(rustcsrc
    PUBLIC "${CMAKE_SOURCE_DIR}/ncore/include"
    PRIVATE "${CMAKE_SOURCE_DIR}/ncore/rust/csrc"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(rustcsrc PRIVATE
    ${COMPILE_FLAGS}
    ${SIMD_FLAGS}
    $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)

nova_configure_cpu_target(rustcsrc)

if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(rustcsrc)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(rustcsrc)
endif()
