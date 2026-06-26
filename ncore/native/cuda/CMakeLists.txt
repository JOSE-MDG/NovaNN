#[=======================================================================[.rst:
CUDA Backend
------------

Build the CUDA backend object library.  Collects all ``.cpp`` source
files (including the ``memory/`` subdirectory) and merges the
``cuda_kernels_obj`` object library into it.

Targets created:

``cuda_obj``
  OBJECT library containing CUDA host-side code and device kernels.

#]=======================================================================]

file(GLOB CUDA_RESOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/ncore/native/cuda/*.cpp"
    "${CMAKE_SOURCE_DIR}/ncore/native/cuda/memory/*.cpp"
)

add_library(cuda_obj OBJECT ${CUDA_RESOURCES})

set_target_properties(cuda_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(cuda_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/rust/csrc
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(kernels)

nova_configure_build_flags(cuda_obj)
nova_configure_cpu_target(cuda_obj)
nova_configure_cuda_runtime_target(cuda_obj)
