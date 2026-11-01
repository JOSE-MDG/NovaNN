#[=======================================================================[.rst:
CUDA Backend
------------

Builds the CUDA host-side object library (``cuda_obj``) and its device
kernels (``cuda_kernels_obj``, see ``kernels/``).
#]=======================================================================]

file(GLOB CUDA_RESOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/*.cpp"
)

add_library(cuda_obj OBJECT ${CUDA_RESOURCES})
add_library(ncore::native::obj::cuda ALIAS cuda_obj)

set_target_properties(cuda_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 23
  CXX_STANDARD_REQUIRED ON
)

target_include_directories(cuda_obj
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
  PRIVATE ${CMAKE_SOURCE_DIR}/ncore
  PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/memory
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(kernels)

nova_configure_build_flags(cuda_obj)
nova_configure_cpu_target(cuda_obj)
nova_configure_cuda_runtime_target(cuda_obj)
