#[=======================================================================[.rst:
Kernels
-------

Device-agnostic kernel dispatch layer, built as the ``kernels_obj``
OBJECT library.  Device-specific kernels live in ``cuda/kernels/``
and ``hip/kernels/`` under their respective backend targets.
#]=======================================================================]

file(GLOB_RECURSE KERNEL_RESOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/ncore/native/kernels/*.cpp"
)

add_library(kernels_obj OBJECT ${KERNEL_RESOURCES})
add_library(ncore::native::obj::kernels ALIAS kernels_obj)

set_target_properties(kernels_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(kernels_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

nova_configure_build_flags(kernels_obj)
nova_configure_cpu_target(kernels_obj)

if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(kernels_obj)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(kernels_obj)
endif()
