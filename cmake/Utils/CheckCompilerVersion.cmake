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

  ``GNU``
    Minimum ``14.0.0``.

  ``Clang``
    Minimum ``17.0.0``.

  ``MSVC``
    Minimum ``19.38`` (VS2022 17.8, the first release with reasonable
    C++23 support). MSVC builds use fewer SIMD optimizations than
    GCC/Clang builds. MSVC is accepted so the project stays buildable
    on Windows.

  NovaNN only supports GCC/G++, Clang/Clang++, and MSVC (``cl.exe``).
  Any other compiler ID (for example ``AppleClang`` or ``IntelLLVM``)
  is rejected.

  On failure the function calls :command:`message` with
  ``FATAL_ERROR``, reporting the compiler path, the detected version,
  and the minimum required version, then halts configuration.

#]=======================================================================]
function(check_min_compiler_version)
  set(_gnu_min "14.0.0")
  set(_clang_min "17.0.0")
  set(_msvc_min "19.38")

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
    elseif(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
      if(CMAKE_${lang}_COMPILER_VERSION VERSION_LESS _msvc_min)
        message(FATAL_ERROR
          "${CMAKE_${lang}_COMPILER} (MSVC ${CMAKE_${lang}_COMPILER_VERSION}) "
          "is too old.  NovaNN requires MSVC ${_msvc_min} or later."
        )
      endif()

      message(STATUS
        "${lang} compiler is MSVC (cl.exe): fewer SIMD optimizations "
        "will be used for this build."
      )
    else()
      message(FATAL_ERROR
        "${CMAKE_${lang}_COMPILER} uses unsupported compiler ID "
        "'${CMAKE_${lang}_COMPILER_ID}'.  NovaNN only supports GCC/G++, "
        "Clang/Clang++ (including clang-cl), and MSVC (cl.exe)."
      )
    endif()
  endforeach()
endfunction()
