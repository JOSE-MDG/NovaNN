#[=======================================================================[.rst:
DetectSanitizers
----------------

Configure AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan)
compile and link options for Nova using modern CMake interface targets.

User Options
^^^^^^^^^^^^

This module consults the following CMake options:

``USE_ASAN``
  Enable AddressSanitizer (ASan) instrumentation.

``USE_UBSAN``
  Enable UndefinedBehaviorSanitizer (UBSan) instrumentation.

If neither option is enabled, the module returns immediately.

Defined Targets
^^^^^^^^^^^^^^^

When enabled, this module creates the following target:

``nova::sanitizers`` (alias for ``nova_sanitizers``)
  An ``INTERFACE`` library containing all compile and link flags required by
  the enabled sanitizers. Consumers should link against it using:

  .. code-block:: cmake

     target_link_libraries(<target> PRIVATE nova::sanitizers)

Cache Variables
^^^^^^^^^^^^^^^

This module sets the following internal cache variables:

``NOVA_HAS_ASAN``
  Set to ``1`` when ``USE_ASAN`` is enabled and configured.

``NOVA_HAS_UBSAN``
  Set to ``1`` when ``USE_UBSAN`` is enabled and configured.

.. note::
  UBSan is not supported by clang-cl and will be skipped with a warning
  when building on Windows (CMake frontend variant ``MSVC``).

.. note::
  When ASan is enabled with clang-cl (MSVC frontend variant), this module
  defines ``_DISABLE_STL_ANNOTATION`` on the consumer target. The MSVC
  STL otherwise emits COFF ``/FAILIFMISMATCH`` directives recording the ASan
  container-annotation state (``annotate_string``, ``annotate_vector`` and
  ``annotate_optional``). clang-cl objects built with ``/fsanitize=address``
  report ``1``, but ``.cu`` objects compiled by nvcc are never
  host-instrumented and report ``0``; lld-link rejects that mix when linking
  ``native.lib`` with ``/WHOLEARCHIVE``. Forcing the value ``0`` everywhere
  keeps host ASan coverage while staying linkable against the uninstrumented
  device objects.

  Also, lld-link does not accept ``/fsanitize=address`` as a link option
  (it is consumed by the clang-cl driver only and lld-link would treat it as
  an input file), so this module links the ASan runtime libraries explicitly
  on the consumer, replicating the clang-cl driver behavior: the import
  library ``clang_rt.asan_dynamic-x86_64.lib`` plus the runtime thunk
  ``clang_rt.asan_dynamic_runtime_thunk-x86_64.lib`` with ``/WHOLEARCHIVE``
  and ``/INCLUDE:__asan_seh_interceptor``. The thunk is required because the
  data globals that instrumented code references
  (``__asan_shadow_memory_dynamic_address``,
  ``__asan_option_detect_stack_use_after_return``) are defined per-module by
  the thunk, not exported by the runtime DLL. The runtime DLL
  (``clang_rt.asan_dynamic-x86_64.dll``) must be available at load time.

#]=======================================================================]

if(NOT USE_ASAN AND NOT USE_UBSAN)
  return()
endif()

set(_san_is_clang_cl FALSE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
   CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(_san_is_clang_cl TRUE)
endif()

if(NOT TARGET nova_sanitizers)
  add_library(nova_sanitizers INTERFACE)
  add_library(nova::sanitizers ALIAS nova_sanitizers)
endif()

# Clang does not link the sanitizer runtime into shared libraries, so
# libnova.so would fail its -Wl,--no-undefined link with undefined
# __asan_* / __ubsan_handle_* references. Resolve the runtime directory
# (-print-runtime-dir) once and reuse it below.
set(_nova_san_runtime_dir "")
set(_nova_san_rpath "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT _san_is_clang_cl AND NOT WIN32)
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-runtime-dir
    OUTPUT_VARIABLE _nova_san_runtime_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_nova_san_runtime_dir AND IS_DIRECTORY "${_nova_san_runtime_dir}")
    set(_nova_san_rpath "-Wl,-rpath,${_nova_san_runtime_dir}")
  else()
    set(_nova_san_runtime_dir "")
    message(WARNING
      "Could not determine Clang's sanitizer runtime directory "
      "(clang -print-runtime-dir failed). Sanitized shared libraries may "
      "fail to link with undefined __ubsan_handle_* / __asan_* references."
    )
  endif()
endif()

# AddressSanitizer
if(USE_ASAN)
  if(_san_is_clang_cl)
    target_compile_options(nova_sanitizers INTERFACE
      /fsanitize=address
      /clang:-fno-omit-frame-pointer
    )
    # `/fsanitize=address` is a clang-cl driver option, not an lld-link one
    # (lld-link would treat it as an input file). Link the runtime explicitly:
    # /MD builds use clang_rt.asan_dynamic-<arch>.lib (see module note).
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
      OUTPUT_VARIABLE _nova_san_resource_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    set(_nova_asan_rt
      "${_nova_san_resource_dir}/lib/windows/clang_rt.asan_dynamic-x86_64.lib")
    set(_nova_asan_thunk
      "${_nova_san_resource_dir}/lib/windows/clang_rt.asan_dynamic_runtime_thunk-x86_64.lib")
    if(EXISTS "${_nova_asan_rt}" AND EXISTS "${_nova_asan_thunk}")
      target_link_libraries(nova_sanitizers INTERFACE "${_nova_asan_rt}")
      # The runtime DLL exports only functions; the per-module data globals
      # that instrumented code references come from the runtime thunk. Mirror
      # clang-cl's own link line: thunk via /WHOLEARCHIVE plus
      # /INCLUDE:__asan_seh_interceptor.
      target_link_options(nova_sanitizers INTERFACE
        "/WHOLEARCHIVE:${_nova_asan_thunk}"
        "/INCLUDE:__asan_seh_interceptor"
      )
    else()
      message(WARNING
        "clang_rt.asan_dynamic-x86_64.lib or "
        "clang_rt.asan_dynamic_runtime_thunk-x86_64.lib not found in "
        "${_nova_san_resource_dir}/lib/windows. ASan-instrumented binaries may "
        "fail to link with undefined __asan_shadow_memory_dynamic_address."
      )
    endif()
    # nvcc never host-instruments .cu objects, so they report STL container
    # annotations disabled while clang-cl objects report enabled; the
    # /WHOLEARCHIVE'd native.lib link then fails the /FAILIFMISMATCH check.
    # Force the disabled state everywhere (see module note).
    target_compile_definitions(nova_sanitizers INTERFACE
      _DISABLE_STL_ANNOTATION
    )
  else()
    set(NOVA_HAS_ASAN 1 CACHE INTERNAL "")
    target_compile_options(nova_sanitizers INTERFACE
      -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(nova_sanitizers INTERFACE
      -fsanitize=address
      # Required for libnova.so to satisfy -Wl,--no-undefined.
      -shared-libasan
      ${_nova_san_rpath}
    )
  endif()
endif()

# UndefinedBehaviorSanitizer
if(USE_UBSAN)
  if(_san_is_clang_cl)
    message(WARNING
      "UBSan (-fsanitize=undefined) is not supported by clang-cl. "
      "NOVA_HAS_UBSAN will not be set. "
      "Rebuild on Linux with native Clang or GCC to use UBSan."
    )
  else()
    set(NOVA_HAS_UBSAN 1 CACHE INTERNAL "")
    # Clang links no UBSan runtime into shared libraries; inject it. The
    # --push-state,--no-as-needed pair keeps it on the line despite
    # -Wl,--as-needed + LTO.
    set(_nova_ubsan_runtime_link "")
    set(_nova_san_arch "x86_64")
    if(_nova_san_arch AND
       EXISTS "${_nova_san_runtime_dir}/libclang_rt.ubsan_standalone-x86_64.so")
      set(_nova_ubsan_runtime_link
        -Wl,--push-state,--no-as-needed
        "${_nova_san_runtime_dir}/libclang_rt.ubsan_standalone-${_nova_san_arch}.so"
        -Wl,--pop-state
      )
    elseif(_nova_san_runtime_dir)
      message(WARNING
        "UBSan runtime libclang_rt.ubsan_standalone not found in "
        "'${_nova_san_runtime_dir}'. Shared library link may fail with "
        "undefined __ubsan_handle_* references."
      )
    endif()
    target_compile_options(nova_sanitizers INTERFACE
      -fsanitize=undefined -fno-sanitize-recover=all)
    target_link_options(nova_sanitizers INTERFACE
      -fsanitize=undefined
      ${_nova_ubsan_runtime_link}
      ${_nova_san_rpath}
    )
  endif()
endif()
