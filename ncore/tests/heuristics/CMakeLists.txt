#[=======================================================================[.rst:
Heuristics Tests
----------------

GoogleTest suite for the kernel launch heuristics
(``ncore/include/ncore/headeronly/heuristics/kernels/``): tier routing,
Fast/Full geometry per pattern, shared clamps, and the advisory Full
config cache. All cases are host-side arithmetic with stub device
caps, so no GPU is required.

Targets created:

``ncore_heuristics_tests`` — alias ``ncore::tests::native::heuristics``
  GoogleTest executable built from all ``*.cpp`` files in this
  directory.
#]=======================================================================]

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if(TEST_SOURCES)
    add_executable(ncore_heuristics_tests ${TEST_SOURCES})
    add_executable(ncore::tests::native::heuristics ALIAS ncore_heuristics_tests)

    target_include_directories(ncore_heuristics_tests
        PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_BINARY_DIR}
    )

    target_link_libraries(ncore_heuristics_tests
        PRIVATE ncore::memory
        PRIVATE ncore::native
        PRIVATE ncore::obj::core
        PRIVATE ncore::obj::dtypes
    )

    nova_configure_gtest_target(ncore_heuristics_tests)
    nova_configure_build_flags(ncore_heuristics_tests)
    nova_configure_cpu_target(ncore_heuristics_tests)

    add_dependencies(ncore_heuristics_tests nova::codegen)

    if(BUILD_TESTING AND NOVA_HAS_GTEST)
        gtest_discover_tests(ncore_heuristics_tests
            DISCOVERY_TIMEOUT 60
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

unset(TEST_SOURCES)
