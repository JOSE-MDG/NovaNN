#[=======================================================================[.rst:
NCore
-----

Core module of the NovaNN project.  Includes the native and Rust
subdirectories, then builds the core object libraries from C and C++
source files.

Targets created:

``ncore_obj`` — alias ``ncore::obj::core``
  OBJECT library containing core C sources (``repr/`` and ``core/``
  subdirectories).

``autograd_obj`` — alias ``ncore::obj::autograd``
  OBJECT library containing autograd C++ sources.

``dtypes_obj`` — alias ``ncore::obj::dtypes``
  OBJECT library containing reduced-precision dtype sources.

The object libraries link against the ``native`` backend (alias
``ncore::native``) and receive build flags, CPU configuration, and
optional CUDA/HIP configuration.

#]=======================================================================]

add_subdirectory(native)
add_subdirectory(rust)

# Create the core object library from C sources in the repr and core subdirectories.
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/ncore/src/repr/*.c"
    "${CMAKE_SOURCE_DIR}/ncore/src/core/*.c"
)

add_library(ncore_obj OBJECT ${CORE_SOURCES})
add_library(ncore::obj::core ALIAS ncore_obj)

set_target_properties(ncore_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 23
    C_STANDARD_REQUIRED ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(ncore_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/src>
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/cuda
    PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/hip
    PRIVATE ${CMAKE_BINARY_DIR}
)

target_link_libraries(ncore_obj PRIVATE ncore::native)

nova_configure_build_flags(ncore_obj)
nova_configure_cpu_target(ncore_obj)

if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(ncore_obj)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(ncore_obj)
endif()

add_dependencies(ncore_obj nova::codegen)

# Autograd object library
file(GLOB_RECURSE AUTOGRAD_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/ncore/src/autograd/*.cpp"
)

add_library(autograd_obj OBJECT ${AUTOGRAD_SOURCES})
add_library(ncore::obj::autograd ALIAS autograd_obj)

set_target_properties(autograd_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(autograd_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_BINARY_DIR}
)

target_link_libraries(autograd_obj PRIVATE ncore::native)

nova_configure_build_flags(autograd_obj)
nova_configure_cpu_target(autograd_obj)

if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(autograd_obj)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(autograd_obj)
endif()

add_dependencies(autograd_obj nova::codegen)

# Custom dtypes object library
file(GLOB DTYPES_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/ncore/src/dtypes/*.cpp"
)

add_library(dtypes_obj OBJECT ${DTYPES_SOURCES})
add_library(ncore::obj::dtypes ALIAS dtypes_obj)

set_target_properties(dtypes_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(dtypes_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
    PRIVATE ${CMAKE_BINARY_DIR}
)

nova_configure_build_flags(dtypes_obj)
nova_configure_cpu_target(dtypes_obj)

if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(dtypes_obj)
    nova_configure_cuda_includes_target(dtypes_obj)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(dtypes_obj)
    nova_configure_hip_includes_target(dtypes_obj)
endif()

add_dependencies(dtypes_obj nova::codegen)
