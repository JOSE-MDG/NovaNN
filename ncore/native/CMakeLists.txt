#[=======================================================================[.rst:
Native Backend
--------------

Aggregates all backend object libraries into a single static archive
``libnative.a``.

All of them must be listed explicitly here via
``$<TARGET_OBJECTS:...>`` so every ``.o`` file is physically written
into the archive: ``target_link_libraries`` between OBJECT libraries
only propagates compile flags and include paths, and
``target_sources($<TARGET_OBJECTS:...>)`` on an OBJECT library is not
reliably expanded by downstream consumers with the Makefile generator.
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

list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:ncore::native::obj::cpu>)

if(TARGET ncore::native::obj::cuda::kernels)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:ncore::native::obj::cuda::kernels>)
endif()

if(TARGET ncore::native::obj::hip::kernels)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:ncore::native::obj::hip::kernels>)
endif()

if(TARGET ncore::native::obj::cuda)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:ncore::native::obj::cuda>)
endif()

if(TARGET ncore::native::obj::hip)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:ncore::native::obj::hip>)
endif()

if(TARGET ncore::native::obj::kernels)
    list(APPEND NATIVE_SOURCES $<TARGET_OBJECTS:ncore::native::obj::kernels>)
endif()

add_library(native STATIC ${NATIVE_SOURCES})
add_library(ncore::native ALIAS native)

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

if(TARGET nova::compiler_rt_builtins)
    target_link_libraries(native INTERFACE nova::compiler_rt_builtins)
endif()
