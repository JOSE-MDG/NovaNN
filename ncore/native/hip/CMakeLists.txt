#[=======================================================================[.rst:
HIP Backend
-----------

Builds the HIP host-side object library (``hip_obj``) and its device
kernels (``hip_kernels_obj``, see ``kernels/``).
#]=======================================================================]

file(GLOB HIP_RESOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/*.cpp"
)

add_library(hip_obj OBJECT ${HIP_RESOURCES})
add_library(ncore::native::obj::hip ALIAS hip_obj)

set_target_properties(hip_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 23
  CXX_STANDARD_REQUIRED ON
)

target_include_directories(hip_obj
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
  PRIVATE ${CMAKE_SOURCE_DIR}/ncore/memory/csrc
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/memory
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(kernels)

nova_configure_build_flags(hip_obj)
nova_configure_cpu_target(hip_obj)
nova_configure_hip_runtime_target(hip_obj)
