#[=======================================================================[.rst:
DTypes Tests
------------

GoogleTest suite for the reduced-precision dtype layer
(``ncore/src/dtypes/DTypes.cpp``): soft-float primitives such as
``fp16_to_float``, ``bf16_to_float`` and the FP4/FP8 encode and decode
helpers, plus their agreement with the scalar casting kernels.

The glob is recursive so the ``bitPatternIdentity/`` subdirectory suites
are picked up alongside the top-level test files.

Targets created:

``ncore_dtypes_tests`` — alias ``ncore::tests::native::dtypes``
  GoogleTest executable built from all ``*.cpp`` files under this
  directory (including subdirectories).
#]=======================================================================]

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if(TEST_SOURCES)
    add_executable(ncore_dtypes_tests ${TEST_SOURCES})
    add_executable(ncore::tests::native::dtypes ALIAS ncore_dtypes_tests)

    target_include_directories(ncore_dtypes_tests
        PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_BINARY_DIR}
    )

    target_link_libraries(ncore_dtypes_tests
        PRIVATE ncore::memory
        PRIVATE ncore::native
        PRIVATE ncore::obj::core
        PRIVATE ncore::obj::dtypes
    )

    nova_configure_gtest_target(ncore_dtypes_tests)
    nova_configure_build_flags(ncore_dtypes_tests)
    nova_configure_cpu_target(ncore_dtypes_tests)
    add_dependencies(ncore_dtypes_tests nova::codegen)

    if(BUILD_TESTING AND NOVA_HAS_GTEST)
        gtest_discover_tests(ncore_dtypes_tests
            DISCOVERY_TIMEOUT 60
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

unset(TEST_SOURCES)
