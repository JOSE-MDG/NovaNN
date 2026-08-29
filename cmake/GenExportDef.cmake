#[=======================================================================[.rst:
GenExportDef
------------

Generates a Windows linker module-definition (``.def``) file listing the
exported symbols of one or more static libraries. On non-Windows
platforms the script returns immediately without doing any work.

This script is intended to be executed with ``cmake -P`` and expects
its input variables to be supplied via ``-D`` on the command line.

This script consumes the following variables:

``INPUT_LIST``
  Path to a file containing a semicolon-separated list of paths to the
  static libraries to scan (as produced by ``file(GENERATE OUTPUT ...)``).

``OUTPUT_DEF``
  Destination path of the generated ``.def`` file. The file starts
  with an ``EXPORTS`` header followed by one exported symbol per line.

Symbol filtering
^^^^^^^^^^^^^^^^

For every library, the script skips:

- Symbols whose name starts with ``.`` (toolchain-local symbols).
- ``__imp_*`` symbols (import thunks generated for imported DLLs).
- ``__NULL_*`` symbols (linker-generated null thunks).
- ``?``-prefixed symbols (C++ names decorated per the MSVC ABI, as
  emitted by clang-cl).

Only C-linkage (undecorated) symbols are exported. Missing or
unreadable library paths are skipped. A missing ``CMAKE_NM`` or a
failed symbol dump for a given library is reported as a warning; the
script continues with the remaining libraries.

#]=======================================================================]

if(NOT WIN32)
  return()
endif()

if(NOT INPUT_LIST)
  message(FATAL_ERROR "GenExportDef: INPUT_LIST is not set")
endif()

if(NOT OUTPUT_DEF)
  message(FATAL_ERROR "GenExportDef: OUTPUT_DEF is not set")
endif()

if(NOT CMAKE_NM)
  message(FATAL_ERROR
    "GenExportDef: CMAKE_NM is not set; pass -DCMAKE_NM=<path-to-nm> "
    "(e.g. llvm-nm) on the command line")
endif()

if(NOT EXISTS "${INPUT_LIST}")
  message(FATAL_ERROR
    "GenExportDef: INPUT_LIST file does not exist: '${INPUT_LIST}'")
endif()

file(READ "${INPUT_LIST}" _nova_scanlist_raw)
string(STRIP "${_nova_scanlist_raw}" INPUT_LIST)

if(NOT INPUT_LIST)
  message(FATAL_ERROR
    "GenExportDef: INPUT_LIST file '${INPUT_LIST}' is empty or malformed")
endif()

set(_nova_exported_symbols "")

foreach(_lib IN LISTS INPUT_LIST)
  if(NOT EXISTS "${_lib}")
    message(WARNING "GenExportDef: skipping missing library '${_lib}'")
    continue()
  endif()

  execute_process(
    COMMAND "${CMAKE_NM}" -g "${_lib}"
    OUTPUT_VARIABLE _nm_output
    ERROR_VARIABLE _nm_error
    RESULT_VARIABLE _nm_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(NOT _nm_result EQUAL 0)
    message(WARNING
      "GenExportDef: '${CMAKE_NM} -g ${_lib}' failed (${_nm_result}): ${_nm_error}")
    continue()
  endif()

  string(REPLACE "\n" ";" _nm_lines "${_nm_output}")

  foreach(_line IN LISTS _nm_lines)
    if(_line MATCHES " [TW] ")
      string(REGEX REPLACE "^.* [TW] " "" _sym "${_line}")
      string(STRIP "${_sym}" _sym)

      if(NOT _sym MATCHES "^\\.|^__imp_|^__NULL|^\\?")
        list(APPEND _nova_exported_symbols "${_sym}")
      endif()
    endif()
  endforeach()
endforeach()

list(REMOVE_DUPLICATES _nova_exported_symbols)

list(JOIN _nova_exported_symbols "\n  " _nova_exports_body)
file(WRITE "${OUTPUT_DEF}" "EXPORTS\n  ${_nova_exports_body}\n")

list(LENGTH _nova_exported_symbols _nova_export_count)
