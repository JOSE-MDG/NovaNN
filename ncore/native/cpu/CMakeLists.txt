#[=======================================================================[.rst:
CPU Backend
-----------

Collects all C source files in the CPU backend directory and builds
the ``cpu_obj`` OBJECT library.  Targets in this subtree provide
vectorised (SIMD) implementations of element-wise operations.

Targets created
^^^^^^^^^^^^^^^

``cpu_obj`` — alias ``ncore::native::obj::cpu``
  OBJECT library containing all CPU backend kernel implementations.

Functions used
^^^^^^^^^^^^^^

:command:`nova_configure_cpu_target`
  Applies SIMD compiler flags, links ``Threads::Threads`` and
  ``OpenMP::OpenMP_C`` when available (defined in
  :module:`NovaNNCPU`).

#]=======================================================================]

file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/ncore/native/cpu/*.c"
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
