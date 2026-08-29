#[=======================================================================[.rst:
DetectOpenMP
------------

Detect OpenMP support. On clang-cl (LLVM's MSVC-compatible frontend) the
LLVM OpenMP runtime (``libomp``) is not linked automatically and the
standard ``FindOpenMP`` module does not handle clang-cl on Windows, so
this module resolves ``libomp`` through a discovery chain.

The winning strategy is reported via ``message(STATUS)`` for
diagnosability. On every other toolchain this module delegates to
``find_package(OpenMP)``.

This module sets the following variables:

``NOVA_HAS_OPENMP``
  ``1`` if OpenMP support was found and configured, ``0`` otherwise.

``NOVA_OPENMP_COMPILE_FLAGS``
  Compiler flags to enable OpenMP (clang-cl fallback path only).

``NOVA_OPENMP_LIB``
  Full path to the resolved OpenMP runtime library (clang-cl fallback
  path only).

``NOVA_LLVM_ROOT`` (cache variable)
  Optional explicit root of an LLVM installation. Overrides automatic
  discovery when set.

Defines the ``nova::openmp`` interface target.

The module is idempotent: if ``NOVA_HAS_OPENMP`` is already defined
the file returns immediately.

.. note::
  On Windows, ``omp.h`` is not shipped in ``<root>/include`` but in the
  compiler resource directory (``<root>/lib/clang/<version>/include``,
  resolved via ``clang-cl -print-resource-dir``), while ``libomp.lib``
  sits in ``<root>/lib``. Candidate roots are probed against both
  locations, and every probe is performed with ``NO_CACHE`` so a failed
  strategy never poisons the next one.
#]=======================================================================]

if(DEFINED NOVA_HAS_OPENMP)
  return()
endif()

set(NOVA_LLVM_ROOT "" CACHE PATH
  "Root of an LLVM installation providing omp.h and lib/libomp.* (clang-cl OpenMP fallback). Leave empty for automatic discovery.")

set(_omp_is_clang_cl FALSE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
   CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(_omp_is_clang_cl TRUE)
endif()

if(NOT TARGET nova_openmp)
  add_library(nova_openmp INTERFACE)
  add_library(nova::openmp ALIAS nova_openmp)
endif()

if(_omp_is_clang_cl)
  # omp.h lives in the compiler resource directory on Windows
  # (<root>/lib/clang/<version>/include). Resolve it once and reuse it as an
  # extra hint for every candidate root.
  set(_omp_resource_include "")
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
    OUTPUT_VARIABLE _omp_resource_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_omp_resource_dir AND IS_DIRECTORY "${_omp_resource_dir}")
    set(_omp_resource_include "${_omp_resource_dir}/include")
  endif()
  unset(_omp_resource_dir)

  # Candidate roots probed for omp.h and libomp. Explicit user configuration
  # always wins over automatic discovery.
  set(_omp_roots "")
  set(_omp_strategies "")
  if(NOVA_LLVM_ROOT)
    list(APPEND _omp_roots "${NOVA_LLVM_ROOT}")
    list(APPEND _omp_strategies "NOVA_LLVM_ROOT cache variable")
  endif()

  if(WIN32)
    # The LLVM installer (a 32-bit NSIS build) writes HKLM\SOFTWARE\LLVM\LLVM
    # into the 32-bit registry view (WOW6432Node), storing the install
    # directory in the key's default value; CMake reads that view for
    # compatibility, while 64-bit reg.exe queries never see it. A missing key
    # expands to "/registry" in modern CMake.
    get_filename_component(_omp_reg_root
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\LLVM\\LLVM]" ABSOLUTE)
    if(NOT _omp_reg_root STREQUAL "/registry" AND
       NOT _omp_reg_root MATCHES "^\\[HKEY")
      list(APPEND _omp_roots "${_omp_reg_root}")
      list(APPEND _omp_strategies "Windows registry (HKLM\\SOFTWARE\\LLVM\\LLVM)")
    endif()
    unset(_omp_reg_root)
  endif()

  foreach(_omp_env_var LLVM_ROOT LLVM_DIR)
    if(DEFINED ENV{${_omp_env_var}})
      list(APPEND _omp_roots "$ENV{${_omp_env_var}}")
      list(APPEND _omp_strategies "environment variable ${_omp_env_var}")
    endif()
  endforeach()

  get_filename_component(_omp_clang_real "${CMAKE_CXX_COMPILER}" REALPATH)
  get_filename_component(_omp_compiler_root "${_omp_clang_real}" DIRECTORY)
  get_filename_component(_omp_compiler_root "${_omp_compiler_root}" DIRECTORY)
  list(APPEND _omp_roots "${_omp_compiler_root}")
  list(APPEND _omp_strategies "path relative to compiler")
  unset(_omp_clang_real)
  unset(_omp_compiler_root)

  set(_omp_include_dir "")
  set(_omp_lib "")
  foreach(_omp_root IN LISTS _omp_roots)
    if(_omp_include_dir AND _omp_lib)
      break()
    endif()
    list(POP_FRONT _omp_strategies _omp_candidate_strategy)
    find_path(_omp_candidate_inc NAMES omp.h
      HINTS "${_omp_root}/include" "${_omp_resource_include}"
      NO_DEFAULT_PATH NO_CACHE)
    find_library(_omp_candidate_lib NAMES libomp omp
      HINTS "${_omp_root}/lib" NO_DEFAULT_PATH NO_CACHE)
    if(_omp_candidate_inc AND _omp_candidate_lib)
      set(_omp_include_dir "${_omp_candidate_inc}")
      set(_omp_lib "${_omp_candidate_lib}")
      set(_omp_strategy "${_omp_candidate_strategy}")
    endif()
    unset(_omp_candidate_inc)
    unset(_omp_candidate_lib)
  endforeach()

  if(NOT (_omp_include_dir AND _omp_lib))
    find_package(LLVM CONFIG QUIET)
    if(LLVM_FOUND)
      find_path(_omp_candidate_inc NAMES omp.h
        HINTS ${LLVM_INCLUDE_DIRS} NO_DEFAULT_PATH NO_CACHE)
      find_library(_omp_candidate_lib NAMES libomp omp
        HINTS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH NO_CACHE)
      if(_omp_candidate_inc AND _omp_candidate_lib)
        set(_omp_include_dir "${_omp_candidate_inc}")
        set(_omp_lib "${_omp_candidate_lib}")
        set(_omp_strategy "LLVM package config")
      endif()
      unset(_omp_candidate_inc)
      unset(_omp_candidate_lib)
    endif()
  endif()

  if(NOT (_omp_include_dir AND _omp_lib))
    find_path(_omp_include_dir NAMES omp.h NO_CACHE)
    find_library(_omp_lib NAMES libomp omp NO_CACHE)
    if(_omp_include_dir AND _omp_lib)
      set(_omp_strategy "system search path")
    endif()
  endif()

  if(_omp_include_dir AND _omp_lib)
    set(NOVA_HAS_OPENMP 1)
    set(NOVA_OPENMP_COMPILE_FLAGS "/clang:-fopenmp")
    set(NOVA_OPENMP_LIB "${_omp_lib}")

    target_compile_options(nova_openmp INTERFACE ${NOVA_OPENMP_COMPILE_FLAGS})
    target_include_directories(nova_openmp INTERFACE "${_omp_include_dir}")
    target_link_libraries(nova_openmp INTERFACE "${NOVA_OPENMP_LIB}")
    message(STATUS
      "Threading: OpenMP found via ${_omp_strategy}: ${NOVA_OPENMP_LIB}")
  else()
    set(NOVA_HAS_OPENMP 0)
    message(STATUS
      "Threading: OpenMP NOT found. Set NOVA_LLVM_ROOT (or the LLVM_ROOT "
      "environment variable) to your LLVM installation if libomp is present "
      "but not being discovered.")
  endif()

  unset(_omp_include_dir)
  unset(_omp_include_dir CACHE)
  unset(_omp_lib)
  unset(_omp_lib CACHE)
  unset(_omp_strategy)
  unset(_omp_resource_include)
else()
  find_package(OpenMP COMPONENTS C CXX)

  if(OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
    set(NOVA_HAS_OPENMP 1)
    target_link_libraries(nova_openmp INTERFACE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    message(STATUS "Threading: OpenMP ${OpenMP_VERSION} found")
  else()
    set(NOVA_HAS_OPENMP 0)
    message(STATUS "Threading: OpenMP NOT found")
  endif()
endif()
