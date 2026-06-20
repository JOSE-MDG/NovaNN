#[=======================================================================[.rst:
CheckInstructionSupport
-----------------------

Utility macro for detecting SIMD instruction set support at configure
time.  This module wraps :command:`check_cxx_source_runs` to test
whether the compiler can emit and execute specific SIMD instructions.

This module defines the following functions:

.. command:: check_simd

  Test whether the compiler supports a given SIMD instruction snippet:

  .. code-block:: cmake

    check_simd(<VAR> <TEST_FLAGS> <APPEND_FLAGS> <SNIPPET>)

  ``<VAR>`
    Variable name to store the test result (set to ``1`` on success).

  ``<TEST_FLAGS>``
    Compiler flags required to enable the instruction set (e.g.,
    ``-mavx2``).

  ``<APPEND_FLAGS>``
    Flags to append to ``SIMD_FLAGS`` when the test passes.

  ``<SNIPPET>``
    C++ source code to compile and run as the detection test.

  On success the macro appends ``<APPEND_FLAGS>`` (split on whitespace)
  to the ``SIMD_FLAGS`` list variable.

#]=======================================================================]

include(CheckCXXSourceRuns)

#[=======================================================================[.rst:
.. command:: check_simd

  Test whether the compiler supports a given SIMD instruction snippet:

  .. code-block:: cmake

    check_simd(<VAR> <TEST_FLAGS> <APPEND_FLAGS> <SNIPPET>)

  ``<VAR>`
    Variable name to store the test result (set to ``1`` on success).

  ``<TEST_FLAGS>``
    Compiler flags required to enable the instruction set (e.g.,
    ``-mavx2``).

  ``<APPEND_FLAGS>``
    Flags to append to ``SIMD_FLAGS`` when the test passes.

  ``<SNIPPET>``
    C++ source code to compile and run as the detection test.

  On success the macro appends ``<APPEND_FLAGS>`` (split on whitespace)
  to the ``SIMD_FLAGS`` list variable.

#]=======================================================================]
macro(check_simd VAR TEST_FLAGS APPEND_FLAGS SNIPPET)
    set(_saved_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${TEST_FLAGS}")
    check_cxx_source_runs("${SNIPPET}" ${VAR})
    set(CMAKE_REQUIRED_FLAGS "${_saved_flags}")

    if(${VAR})
        separate_arguments(_simd_flags UNIX_COMMAND "${APPEND_FLAGS}")
        list(APPEND SIMD_FLAGS ${_simd_flags})
    endif()
endmacro()
