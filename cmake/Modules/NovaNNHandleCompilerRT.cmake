#[=======================================================================[.rst:
NovaNNHandleCompilerRT
----------------------

Detect the Clang compiler-rt builtins library and expose it as a CMake
IMPORTED target.  On Windows with clang-cl, the compiler does NOT link
compiler-rt builtins automatically (unlike clang/clang++ on Linux).
Targets that use compiler builtins such as ``__truncdfbf2`` (bfloat16
conversion) or ``__udivti3`` (128-bit integer division) will fail to
link without this library.

This module is a no-op on non-Windows platforms or when the compiler
is not Clang.

Defined Targets
^^^^^^^^^^^^^^^

``nova::compiler_rt_builtins`` (alias for ``nova_compiler_rt_builtins``)
  An ``IMPORTED STATIC`` library pointing to ``clang_rt.builtins-<arch>.lib``.
  Consumers should link against it using:

  .. code-block:: cmake

     target_link_libraries(<target> PRIVATE nova::compiler_rt_builtins)

  On non-Windows or non-Clang compilers, this target is not created.
  Consumers should guard usage with ``if(TARGET nova::compiler_rt_builtins)``.

Cache Variables
^^^^^^^^^^^^^^^

``NOVA_HAS_COMPILER_RT_BUILTINS``
  Set to ``1`` when the builtins library is found, ``0`` otherwise.

#]=======================================================================]

# Guard: only on Windows with Clang compiler
if(NOT WIN32 OR NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(NOVA_HAS_COMPILER_RT_BUILTINS 0 CACHE INTERNAL "")
    return()
endif()

# Query the compiler for its resource directory
execute_process(
    COMMAND ${CMAKE_C_COMPILER} -print-resource-dir
    OUTPUT_VARIABLE _nova_clang_resource_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _nova_clang_resource_dir_rc
)

if(NOT _nova_clang_resource_dir_rc EQUAL 0 OR NOT _nova_clang_resource_dir)
    set(NOVA_HAS_COMPILER_RT_BUILTINS 0 CACHE INTERNAL "")
    message(WARNING
        "Failed to query clang resource directory. "
        "compiler-rt builtins detection skipped."
    )
    return()
endif()

# Construct the path to the builtins library
set(_nova_clang_rt_builtins_path
    "${_nova_clang_resource_dir}/lib/windows/clang_rt.builtins-x86_64.lib"
)

if(EXISTS "${_nova_clang_rt_builtins_path}")
    add_library(nova_compiler_rt_builtins STATIC IMPORTED GLOBAL)
    set_target_properties(nova_compiler_rt_builtins PROPERTIES
        IMPORTED_LOCATION "${_nova_clang_rt_builtins_path}"
    )
    add_library(nova::compiler_rt_builtins ALIAS nova_compiler_rt_builtins)

    set(NOVA_HAS_COMPILER_RT_BUILTINS 1 CACHE INTERNAL "")
    message(STATUS "Found compiler-rt builtins: ${_nova_clang_rt_builtins_path}")
else()
    set(NOVA_HAS_COMPILER_RT_BUILTINS 0 CACHE INTERNAL "")
    message(WARNING
        "compiler-rt builtins not found at: ${_nova_clang_rt_builtins_path}\n"
        "Targets using compiler builtins (e.g. __truncdfbf2) will fail to link."
    )
endif()

unset(_nova_clang_resource_dir)
unset(_nova_clang_resource_dir_rc)
unset(_nova_clang_rt_builtins_path)
