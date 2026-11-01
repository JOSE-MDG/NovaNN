#[=======================================================================[.rst:
Core Tests
----------

GoogleTest suite for the core runtime (``ncore/src/core/``): status code
propagation, device detection, SIMD capability flags, the Rust-backed
memory allocator, per-device GPU memory pools, and stream-aware tensor layouts.

The executable is only created when ``*.cpp`` files exist, so the file
stays valid even if the directory is emptied.

Targets created:

``ncore_core_tests`` — alias ``ncore::tests::native::core``
  GoogleTest executable built from all ``*.cpp`` files in this
  directory.
#]=======================================================================]

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if(TEST_SOURCES)
    add_executable(ncore_core_tests ${TEST_SOURCES})
    add_executable(ncore::tests::native::core ALIAS ncore_core_tests)

    target_include_directories(ncore_core_tests
        PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_BINARY_DIR}
    )

    target_link_libraries(ncore_core_tests
        PRIVATE ncore::memory
        PRIVATE ncore::native
        PRIVATE ncore::obj::core
        PRIVATE ncore::obj::dtypes
    )

    nova_configure_gtest_target(ncore_core_tests)
    nova_configure_build_flags(ncore_core_tests)
    nova_configure_cpu_target(ncore_core_tests)


if(NOVA_HAS_CUDA)
    nova_configure_cuda_target(ncore_core_tests)
    nova_configure_cuda_runtime_target(ncore_core_tests)
endif()

if(NOVA_HAS_HIP)
    nova_configure_hip_target(ncore_core_tests)
    nova_configure_hip_includes_target(ncore_core_tests)
    nova_configure_hip_runtime_target(ncore_core_tests)
endif()

    add_dependencies(ncore_core_tests nova::codegen)

    if(BUILD_TESTING AND NOVA_HAS_GTEST)
        gtest_discover_tests(ncore_core_tests
            DISCOVERY_TIMEOUT 60
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

unset(TEST_SOURCES)
