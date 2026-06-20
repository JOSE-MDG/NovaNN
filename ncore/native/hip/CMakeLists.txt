#[=======================================================================[.rst:
HIP Backend
-----------

Build the HIP backend object library.  Collects all ``.cpp`` source
files (including the ``memory/`` subdirectory) and merges the
``hip_kernels_obj`` object library into it.

Targets created:

``hip_obj``
  OBJECT library containing HIP host-side code and device kernels.

#]=======================================================================]

file(GLOB_RECURSE HIP_RESOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/ncore/native/hip/*.cpp"
    "${CMAKE_SOURCE_DIR}/ncore/native/hip/memory/*.cpp"
)

add_library(hip_obj OBJECT ${HIP_RESOURCES})

set_target_properties(hip_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(hip_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/rust/csrc
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(kernels)

nova_configure_build_flags(hip_obj)
nova_configure_cpu_target(hip_obj)
nova_configure_hip_runtime_target(hip_obj)
