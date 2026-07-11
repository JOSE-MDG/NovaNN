#[=======================================================================[.rst:
CheckInstructionSupport
-----------------------

Utility macro for detecting SIMD instruction set support at configure
time.  This module wraps :command:`check_cxx_source_runs` to test
whether the compiler can emit and execute specific SIMD instructions.

.. note::
  This macro is a no-op under MSVC. NovaNN's hand-written SIMD kernels
  use ``[[{gnu,clang}::target(...)]]`` to select ISA per function, which is a
  GCC/Clang-only attribute.

This module defines the following functions:

.. command:: check_simd

  Test whether the compiler supports a given SIMD instruction snippet:

  .. code-block:: cmake

    check_simd(<VAR> <TEST_FLAGS> <APPEND_FLAGS> <SNIPPET>)

  ``<VAR>``
    Variable name to store the test result (set to ``1`` on success,
    ``0`` unconditionally under MSVC without running any test).

  ``<TEST_FLAGS>``
    Compiler flags required to enable the instruction set (e.g.,
    ``-mavx2``). Ignored under MSVC.

  ``<APPEND_FLAGS>``
    Flags to append to ``SIMD_FLAGS`` when the test passes. Never
    appended under MSVC.

  ``<SNIPPET>``
    C++ source code to compile and run as the detection test.

  On success the macro appends ``<APPEND_FLAGS>`` (split on whitespace)
  to the ``SIMD_FLAGS`` list variable.

#]=======================================================================]

include(CheckCXXSourceRuns)

macro(check_simd VAR TEST_FLAGS APPEND_FLAGS SNIPPET)
  if(MSVC)
    # See module-level note above: MSVC never gets SIMD kernel flags,
    # so detection is intentionally skipped rather than attempted with
    # GNU-style -m flags that cl.exe would reject or misinterpret.
    set(${VAR} 0)
  else()
    set(_saved_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${TEST_FLAGS}")
    check_cxx_source_runs("${SNIPPET}" ${VAR})
    set(CMAKE_REQUIRED_FLAGS "${_saved_flags}")

    if(${VAR})
      separate_arguments(_simd_flags UNIX_COMMAND "${APPEND_FLAGS}")
      list(APPEND SIMD_FLAGS ${_simd_flags})
    endif()
  endif()
endmacro()
