#[=======================================================================[.rst:
Native Backend
--------------

Aggregates all native backend object libraries into a single static
archive ``libnative.a``.

Object collection
^^^^^^^^^^^^^^^^^

Each backend produces one or more OBJECT libraries.  All of them must
be listed explicitly here via ``$<TARGET_OBJECTS:...>`` so that every
``.o`` file is physically written into the archive.

``target_link_libraries`` between OBJECT libraries only propagates
compile flags and include paths — it does not embed object files.
``target_sources($<TARGET_OBJECTS:...>)`` on an OBJECT library is not
reliably expanded by downstream consumers when using the Makefile
generator.  Explicit enumeration at this level is therefore the only
portable, generator-agnostic approach.

Targets created
^^^^^^^^^^^^^^^

``native``
  STATIC library containing all active backend object files.

Subdirectories
^^^^^^^^^^^^^^

``cpu/``
  Vectorized CPU implementations. Defines cpu_obj

``kernels/``
  CPU-side dispatch.  Always included. Defines kernels_obj

``cuda/``
  CUDA host-side code and device kernels.  Included when
  ``NOVA_HAS_CUDA`` is set.  Defines ``cuda_obj`` and
  ``cuda_kernels_obj``.

``hip/``
  HIP host-side code and device kernels.  Included when
  ``NOVA_HAS_HIP`` is set.  Defines ``hip_obj`` and
  ``hip_kernels_obj``.

#]=======================================================================]

set(NATIVE_SOURCES "")

add_subdirectory(kernels)
add_subdirectory(cpu)

if(NOVA_HAS_CUDA)
    add_subdirectory(cuda)
endif()

if(NOVA_HAS_HIP)
    add_subdirectory(hip)
endif()

if(TARGET cpu_obj)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:cpu_obj>)
endif()

if(TARGET cuda_kernels_obj)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:cuda_kernels_obj>)
endif()

if(TARGET hip_kernels_obj)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:hip_kernels_obj>)
endif()

if(TARGET cuda_obj)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:cuda_obj>)
endif()

if(TARGET hip_obj)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:hip_obj>)
endif()

if(TARGET kernels_obj)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:kernels_obj>)
endif()

if(NOT NATIVE_SOURCES)
    set(NATIVE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/native_stub.c")
endif()

add_library(native STATIC ${NATIVE_SOURCES})

set_target_properties(native PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME native
)

if(NOVA_HAS_CUDA)
    set_target_properties(native PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(NOVA_HAS_HIP)
    set_target_properties(native PROPERTIES
        HIP_STANDARD 23
        HIP_STANDARD_REQUIRED ON
    )
endif()

target_include_directories(native
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
)
