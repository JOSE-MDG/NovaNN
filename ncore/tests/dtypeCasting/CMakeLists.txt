#[=======================================================================[.rst:
DType Casting Tests
-------------------

GoogleTest suite for the dtype casting pipeline: soft-float oracle
utilities (``utils/Oracle.hpp``), cast table metadata and dispatch over
the generated lookup tables, scalar casting kernels and their SIMD
variants.

Current suites: ``Dispatch_test.cpp`` (dispatch matrix and metadata),
``ScalarOracle_test.cpp`` (scalar kernels vs. the independent oracle),
``SaturationInvariant_test.cpp`` (saturation and int conversion
policies), ``SpecialValueInvariant_test.cpp`` (NaN/Inf/zero/subnormal
propagation) and ``ISAEquivalence_test.cpp`` (SIMD variant equivalence).

The executable is only created when ``*.cpp`` files exist, so this file
stays valid even if the directory is emptied.

Targets created:

``ncore_dtype_casting_tests`` — alias ``ncore::tests::native::dtype_casting``
  GoogleTest executable built from all ``*.cpp`` files in this
  directory.
#]=======================================================================]

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if(TEST_SOURCES)
    add_executable(ncore_dtype_casting_tests ${TEST_SOURCES})
    add_executable(ncore::tests::native::dtype_casting ALIAS ncore_dtype_casting_tests)

    target_include_directories(ncore_dtype_casting_tests
        PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ncore/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_BINARY_DIR}
    )

    target_link_libraries(ncore_dtype_casting_tests
        PRIVATE ncore::memory
        PRIVATE ncore::native
        PRIVATE ncore::obj::core
        PRIVATE ncore::obj::dtypes
    )

    nova_configure_gtest_target(ncore_dtype_casting_tests)
    nova_configure_build_flags(ncore_dtype_casting_tests)
    nova_configure_cpu_target(ncore_dtype_casting_tests)
    add_dependencies(ncore_dtype_casting_tests nova::codegen)

    if(BUILD_TESTING AND NOVA_HAS_GTEST)
        gtest_discover_tests(ncore_dtype_casting_tests
            DISCOVERY_TIMEOUT 60
            DISCOVERY_MODE PRE_TEST
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

unset(TEST_SOURCES)
