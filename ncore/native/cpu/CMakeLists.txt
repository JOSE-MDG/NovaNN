#[=======================================================================[.rst:
CPU Backend
-----------

Builds the ``cpu_obj`` OBJECT library from the CPU backend C sources:
vectorised (SIMD) implementations of the CPU kernels.
#]=======================================================================]

file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/ncore/native/cpu/dtype/DTypeCasting.c"
    "${CMAKE_SOURCE_DIR}/ncore/native/cpu/layout/*.c"
)

add_library(cpu_obj OBJECT ${CPU_SOURCES})
add_library(ncore::native::obj::cpu ALIAS cpu_obj)

set_target_properties(cpu_obj PROPERTIES
    C_STANDARD 23
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cpu_obj
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
)

nova_configure_build_flags(cpu_obj)
nova_configure_cpu_target(cpu_obj)

add_dependencies(cpu_obj nova::codegen)
