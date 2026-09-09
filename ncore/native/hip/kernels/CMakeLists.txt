#[=======================================================================[.rst:
HIP Kernels
-----------

Builds the HIP device-code object library.

All ``.hip`` files under this directory are compiled into a single
OBJECT library.  The resulting objects are collected by
``native/CMakeLists.txt`` via ``$<TARGET_OBJECTS:ncore::native::obj::hip::kernels>``
and archived into ``libnative.a``.

Targets created
^^^^^^^^^^^^^^^

``hip_kernels_obj`` — alias ``ncore::native::obj::hip::kernels``
  OBJECT library containing compiled HIP device code.

#]=======================================================================]

file(GLOB_RECURSE KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hip"
)

add_library(hip_kernels_obj OBJECT ${KERNEL_SOURCES})
add_library(ncore::native::obj::hip::kernels ALIAS hip_kernels_obj)

set_target_properties(hip_kernels_obj PROPERTIES
    LINKER_LANGUAGE HIP
    POSITION_INDEPENDENT_CODE ON
    HIP_STANDARD 23
    HIP_STANDARD_REQUIRED ON
)

target_include_directories(hip_kernels_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/memory/csrc
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/hip/memory
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/hip
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

nova_configure_hip_kernels_target(hip_kernels_obj)
