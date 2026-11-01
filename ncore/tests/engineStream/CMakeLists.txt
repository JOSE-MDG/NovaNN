#[=======================================================================[.rst:
Engine Stream Tests
-------------------

GoogleTest suite for the engine-stream pattern: stream reuse, pageable
pipeline correctness, cross-stream reuse guards, pool destroy rejection,
and the rocclr null-stream orphan behavior.

The batched-vs-sync-per-op ratio is a measurement of the host, not of the
code, so it ships as a disabled test that CTest never schedules.

The executable is only created when *.cpp files exist, so the file stays
valid even if the directory is emptied.

Targets created:
  ``ncore_engine_stream_tests`` (alias ``ncore::tests::native::engine_stream``)
#]=======================================================================]

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if(TEST_SOURCES)
    add_executable(ncore_engine_stream_tests ${TEST_SOURCES})
    add_executable(ncore::tests::native::engine_stream ALIAS ncore_engine_stream_tests)

    target_include_directories(ncore_engine_stream_tests
        PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_BINARY_DIR}
    )

    target_link_libraries(ncore_engine_stream_tests
        PRIVATE ncore::memory
        PRIVATE ncore::native
        PRIVATE ncore::obj::core
        PRIVATE ncore::obj::dtypes
    )

    # Device kernels live in a separate object library so the host warning
    # flags from nova_configure_build_flags do not reach nvcc/hipcc.
    if(NOVA_HAS_CUDA)
        add_library(ncore_engine_stream_cuda_kernels_obj OBJECT
            "${CMAKE_CURRENT_SOURCE_DIR}/HazardKernel.cu")
        set_target_properties(ncore_engine_stream_cuda_kernels_obj PROPERTIES
            LINKER_LANGUAGE CUDA
            POSITION_INDEPENDENT_CODE ON
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )
        target_include_directories(ncore_engine_stream_cuda_kernels_obj
            PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
            PRIVATE ${CMAKE_BINARY_DIR}
            PRIVATE ${CMAKE_SOURCE_DIR}/ncore
            PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native
            PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/cuda
            PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/cuda/memory
        )
        nova_configure_cuda_kernels_target(ncore_engine_stream_cuda_kernels_obj)
        target_link_libraries(ncore_engine_stream_tests
            PRIVATE ncore_engine_stream_cuda_kernels_obj)
    endif()

    if(NOVA_HAS_HIP)
        add_library(ncore_engine_stream_hip_kernels_obj OBJECT
            "${CMAKE_CURRENT_SOURCE_DIR}/HazardKernel.hip")
        set_target_properties(ncore_engine_stream_hip_kernels_obj PROPERTIES
            LINKER_LANGUAGE HIP
            POSITION_INDEPENDENT_CODE ON
            HIP_STANDARD 23
            HIP_STANDARD_REQUIRED ON
        )
        target_include_directories(ncore_engine_stream_hip_kernels_obj
            PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
            PRIVATE ${CMAKE_BINARY_DIR}
            PRIVATE ${CMAKE_SOURCE_DIR}/ncore
            PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native
            PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/hip
            PRIVATE ${CMAKE_SOURCE_DIR}/ncore/native/hip/memory
        )
        nova_configure_hip_kernels_target(ncore_engine_stream_hip_kernels_obj)
        target_link_libraries(ncore_engine_stream_tests
            PRIVATE ncore_engine_stream_hip_kernels_obj)
    endif()

    nova_configure_gtest_target(ncore_engine_stream_tests)
    nova_configure_build_flags(ncore_engine_stream_tests)
    nova_configure_cpu_target(ncore_engine_stream_tests)

    if(NOVA_HAS_CUDA)
        nova_configure_cuda_target(ncore_engine_stream_tests)
        nova_configure_cuda_runtime_target(ncore_engine_stream_tests)
    endif()

    if(NOVA_HAS_HIP)
        nova_configure_hip_target(ncore_engine_stream_tests)
        nova_configure_hip_runtime_target(ncore_engine_stream_tests)
        nova_configure_hip_includes_target(ncore_engine_stream_tests)
    endif()

    add_dependencies(ncore_engine_stream_tests nova::codegen)

    if(BUILD_TESTING AND NOVA_HAS_GTEST)
        gtest_discover_tests(ncore_engine_stream_tests
            DISCOVERY_TIMEOUT 60
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

unset(TEST_SOURCES)
