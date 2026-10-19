#[=======================================================================[.rst:
CheckCompilerVersion
--------------------

Verify that the C and C++ compilers meet the minimum versions required
by NovaNN (GCC, Clang, clang-cl only; any other compiler ID is
rejected).
#]=======================================================================]

#[=======================================================================[.rst:
.. command:: check_min_compiler_version

  Enforce the minimum compiler version for the active C and C++
  compilers:

  .. code-block:: cmake

    check_min_compiler_version()

  ``GNU``
    Minimum ``15.0.0``.

  ``Clang``
    Minimum ``20.1.0`` (including clang-cl on Windows).

#]=======================================================================]
function(check_min_compiler_version)
  set(_gnu_min "15.0.0")
  set(_clang_min "20.1.0")

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
          "is too old.  NovaNN requires Clang/Clang++ ${_clang_min} or later "
          "(clang-cl included)."
        )
      endif()
    else()
      message(FATAL_ERROR
        "${CMAKE_${lang}_COMPILER} uses unsupported compiler ID "
        "'${CMAKE_${lang}_COMPILER_ID}'.  NovaNN only supports GCC/G++ "
        "and Clang/Clang++ (including clang-cl on Windows)."
      )
    endif()
  endforeach()
endfunction()
