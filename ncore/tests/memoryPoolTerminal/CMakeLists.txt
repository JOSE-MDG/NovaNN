#[=======================================================================[.rst:
Memory Pool Terminal-State Tests
--------------------------------

Tests the terminal state of a device memory pool after destruction. This
suite has its own executable because a successful pool destroy permanently
marks the process-local device slot as unavailable.

Targets created:

``ncore_memory_pool_terminal_tests``
  GoogleTest executable containing only the isolated terminal-state test.
#]=======================================================================]

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if(TEST_SOURCES)
    add_executable(ncore_memory_pool_terminal_tests ${TEST_SOURCES})
    add_executable(ncore::tests::native::memory_pool_terminal ALIAS ncore_memory_pool_terminal_tests)

    target_include_directories(ncore_memory_pool_terminal_tests
        PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_BINARY_DIR}
    )

    target_link_libraries(ncore_memory_pool_terminal_tests
        PRIVATE ncore::memory
        PRIVATE ncore::native
        PRIVATE ncore::obj::core
        PRIVATE ncore::obj::dtypes
    )

    nova_configure_gtest_target(ncore_memory_pool_terminal_tests)
    nova_configure_build_flags(ncore_memory_pool_terminal_tests)
    nova_configure_cpu_target(ncore_memory_pool_terminal_tests)

    if(NOVA_HAS_CUDA)
        nova_configure_cuda_target(ncore_memory_pool_terminal_tests)
        nova_configure_cuda_runtime_target(ncore_memory_pool_terminal_tests)
    endif()

    if(NOVA_HAS_HIP)
        nova_configure_hip_target(ncore_memory_pool_terminal_tests)
        nova_configure_hip_includes_target(ncore_memory_pool_terminal_tests)
        nova_configure_hip_runtime_target(ncore_memory_pool_terminal_tests)
    endif()

    add_dependencies(ncore_memory_pool_terminal_tests nova::codegen)

    if(BUILD_TESTING AND NOVA_HAS_GTEST)
        gtest_discover_tests(ncore_memory_pool_terminal_tests
            DISCOVERY_TIMEOUT 60
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

unset(TEST_SOURCES)
