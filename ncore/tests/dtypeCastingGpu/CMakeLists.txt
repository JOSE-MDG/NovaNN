#[=======================================================================[.rst:
GPU DType Casting Tests
-----------------------

GoogleTest suite for the on-device dtype casting path: the same 210
dtype pairs as the CPU ``dtypeCasting`` suite, converted on the active
GPU backend (CUDA or HIP) and compared bitwise against the CPU
reference after a device-to-host transfer.

Current suites: ``GpuDispatch_test.cpp`` (pair resolution, device
smoke values, shape and status edges), ``GpuOracleEquivalence_test.cpp``
(full 210-pair bitwise equivalence), ``GpuSaturation_test.cpp``
(saturation and integer conversion policies) and
``GpuSpecialValues_test.cpp`` (NaN/Inf/zero/subnormal propagation).

CPU-only runners skip device cases with a reason string instead of
failing. Edge cases that need no device (shape mismatch, empty
tensors, unsupported pairs through ``cast()``) run unconditionally.

The executable is only created when ``*.cpp`` files exist, so this file
stays valid even if the directory is emptied.

Targets created:

``ncore_dtype_casting_gpu_tests`` — alias ``ncore::tests::native::dtype_casting_gpu``
  GoogleTest executable built from all ``*.cpp`` files in this
  directory.

#]=======================================================================]

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if(TEST_SOURCES)
    add_executable(ncore_dtype_casting_gpu_tests ${TEST_SOURCES})
    add_executable(ncore::tests::native::dtype_casting_gpu ALIAS ncore_dtype_casting_gpu_tests)

    target_include_directories(ncore_dtype_casting_gpu_tests
        PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_BINARY_DIR}
        # Reuse the CPU oracle fixtures and the GPU skip macro instead of
        # duplicating them here.
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../dtypeCasting
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../core
    )

    target_link_libraries(ncore_dtype_casting_gpu_tests
        PRIVATE ncore::memory
        PRIVATE ncore::native
        PRIVATE ncore::obj::core
        PRIVATE ncore::obj::dtypes
    )

    nova_configure_gtest_target(ncore_dtype_casting_gpu_tests)
    nova_configure_build_flags(ncore_dtype_casting_gpu_tests)
    nova_configure_cpu_target(ncore_dtype_casting_gpu_tests)

    if(NOVA_HAS_CUDA)
        nova_configure_cuda_target(ncore_dtype_casting_gpu_tests)
    endif()

    if(NOVA_HAS_HIP)
        nova_configure_hip_target(ncore_dtype_casting_gpu_tests)
    endif()

    add_dependencies(ncore_dtype_casting_gpu_tests nova::codegen)

    if(BUILD_TESTING AND NOVA_HAS_GTEST)
        gtest_discover_tests(ncore_dtype_casting_gpu_tests
            DISCOVERY_TIMEOUT 60
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

unset(TEST_SOURCES)
