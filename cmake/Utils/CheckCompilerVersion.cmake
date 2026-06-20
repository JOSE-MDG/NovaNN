#[=======================================================================[.rst:
CheckCompilerVersion
--------------------

Utility module for verifying that the C and C++ compilers meet the
minimum versions required by NovaNN.

This module defines the following functions:

.. command:: check_min_compiler_version

  Enforce the minimum compiler version for the active C and C++
  compilers:

  .. code-block:: cmake

    check_min_compiler_version()

#]=======================================================================]

#[=======================================================================[.rst:
.. command:: check_min_compiler_version

  Enforce the minimum C and C++ compiler versions required by NovaNN:

  .. code-block:: cmake

    check_min_compiler_version()

  This function must be called after :command:`project` (so that
  :variable:`CMAKE_C_COMPILER_ID`, :variable:`CMAKE_CXX_COMPILER_ID`,
  :variable:`CMAKE_C_COMPILER_VERSION`, and
  :variable:`CMAKE_CXX_COMPILER_VERSION` are populated).  It compares
  the detected compiler family and version against the minimum
  required for each language:

  ``GNU``
    Minimum ``14.0.0``.

  ``Clang``
    Minimum ``17.0.0``.

  NovaNN only supports GCC/G++ and Clang/Clang++.  Any other compiler
  ID (for example ``AppleClang`` or ``IntelLLVM``) is rejected.

  On failure the function calls :command:`message` with
  ``FATAL_ERROR``, reporting the compiler path, the detected version,
  and the minimum required version, then halts configuration.

#]=======================================================================]
function(check_min_compiler_version)
  set(_gnu_min "14.0.0")
  set(_clang_min "17.0.0")

  foreach(lang C CXX)
    if(CMAKE_${lang}_COMPILER_ID STREQUAL "GNU")
      if(CMAKE_${lang}_COMPILER_VERSION VERSION_LESS _gnu_min)
        message(FATAL_ERROR
          "${CMAKE_${lang}_COMPILER} (GNU ${CMAKE_${lang}_COMPILER_VERSION}) "
          "is too old.  NovaNN requires GCC/G++ ${_gnu_min} or later."
        )
      endif()
    elseif(CMAKE_${lang}_COMPILER_ID STREQUAL "Clang")
      if(CMAKE_${lang}_COMPILER_VERSION VERSION_LESS _clang_min)
        message(FATAL_ERROR
          "${CMAKE_${lang}_COMPILER} (Clang ${CMAKE_${lang}_COMPILER_VERSION}) "
          "is too old.  NovaNN requires Clang/Clang++ ${_clang_min} or later."
        )
      endif()
    else()
      message(FATAL_ERROR
        "${CMAKE_${lang}_COMPILER} uses unsupported compiler ID "
        "'${CMAKE_${lang}_COMPILER_ID}'.  NovaNN only supports GCC/G++ "
        "and Clang/Clang++."
      )
    endif()
  endforeach()
endfunction()
