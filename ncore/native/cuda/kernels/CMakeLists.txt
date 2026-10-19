#[=======================================================================[.rst:
CUDA Kernels
------------

Object library for the CUDA device code (``*.cu``).  Its objects are
collected by ``native/CMakeLists.txt`` via
``$<TARGET_OBJECTS:ncore::native::obj::cuda::kernels>`` and archived
into ``libnative.a``.
#]=======================================================================]

file(GLOB_RECURSE KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

add_library(cuda_kernels_obj OBJECT ${KERNEL_SOURCES})
add_library(ncore::native::obj::cuda::kernels ALIAS cuda_kernels_obj)

set_target_properties(cuda_kernels_obj PROPERTIES
    LINKER_LANGUAGE CUDA
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

target_include_directories(cuda_kernels_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_BINARY_DIR}
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/memory/csrc
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/cuda/memory
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/cuda
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

nova_configure_cuda_kernels_target(cuda_kernels_obj)
