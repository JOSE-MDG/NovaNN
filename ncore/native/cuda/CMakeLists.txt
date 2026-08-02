#[=======================================================================[.rst:
CUDA Backend
------------

Builds the CUDA host-side object library and its device kernels.

Targets created
^^^^^^^^^^^^^^^

``cuda_obj`` — alias ``ncore::native::obj::cuda``
  OBJECT library containing all CUDA host-side translation units
  (``*.cpp``, ``memory/*.cpp``).

``cuda_kernels_obj`` — alias ``ncore::native::obj::cuda::kernels``
  (via ``add_subdirectory(kernels)``)
  OBJECT library containing compiled CUDA device code (``*.cu``).
  Collected by ``native/CMakeLists.txt`` via
  ``$<TARGET_OBJECTS:ncore::native::obj::cuda::kernels>``.

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
  PRIVATE ${CMAKE_SOURCE_DIR}/ncore/rust/csrc
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/memory
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(kernels)

nova_configure_build_flags(cuda_obj)
nova_configure_cpu_target(cuda_obj)
nova_configure_cuda_runtime_target(cuda_obj)
