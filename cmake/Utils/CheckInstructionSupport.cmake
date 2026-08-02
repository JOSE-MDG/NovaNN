#[=======================================================================[.rst:
CheckInstructionSupport
-----------------------

Utility macro for detecting SIMD instruction set support at configure
time.  This module wraps :command:`check_cxx_source_runs` to test
whether the compiler can emit and execute specific SIMD instructions.

.. note::
  On clang-cl, GNU-style ``-m`` flags are wrapped with ``/clang:``
  prefix internally.

This module defines the following functions:

.. command:: check_simd

  Test whether the compiler supports a given SIMD instruction snippet:

  .. code-block:: cmake

    check_simd(<VAR> <TEST_FLAGS> <APPEND_FLAGS> <SNIPPET>)

  ``<VAR>``
    Variable name to store the test result (set to ``1`` on success,
    ``0`` on failure).

  ``<TEST_FLAGS>``
    Compiler flags required to enable the instruction set (e.g.,
    ``-mavx2``). On clang-cl these are automatically wrapped with
    ``/clang:`` prefix.

  ``<APPEND_FLAGS>``
    Flags to append to ``SIMD_FLAGS`` when the test passes (also
    wrapped with ``/clang:`` on clang-cl).

  ``<SNIPPET>``
    C++ source code to compile and run as the detection test.

  On success the macro appends ``<APPEND_FLAGS>`` (split on whitespace)
  to the ``SIMD_FLAGS`` list variable.

#]=======================================================================]

include(CheckCXXSourceRuns)

macro(check_simd VAR TEST_FLAGS APPEND_FLAGS SNIPPET)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR
      "NovaNN no longer supports MSVC (cl.exe). "
      "Use clang-cl on Windows."
    )
  elseif(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # clang-cl — wrap GNU-style -m flags with /clang: prefix.
    separate_arguments(_test_flags_list UNIX_COMMAND "${TEST_FLAGS}")
    set(_wrapped "")
    foreach(_flag IN LISTS _test_flags_list)
      list(APPEND _wrapped "/clang:${_flag}")
    endforeach()
    list(JOIN _wrapped " " _wrapped_str)
    set(_saved_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${_wrapped_str}")
    check_cxx_source_runs("${SNIPPET}" ${VAR})
    set(CMAKE_REQUIRED_FLAGS "${_saved_flags}")
    if(${VAR})
      separate_arguments(_append_flags UNIX_COMMAND "${APPEND_FLAGS}")
      set(_wrapped_append "")
      foreach(_flag IN LISTS _append_flags)
        list(APPEND _wrapped_append "/clang:${_flag}")
      endforeach()
      list(APPEND SIMD_FLAGS ${_wrapped_append})
    endif()
    unset(_wrapped_str)
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
