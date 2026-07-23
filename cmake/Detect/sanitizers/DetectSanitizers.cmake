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

#]=======================================================================]

if(NOT USE_ASAN AND NOT USE_UBSAN)
  return()
endif()

if(NOT TARGET nova_sanitizers)
  add_library(nova_sanitizers INTERFACE)
  add_library(nova::sanitizers ALIAS nova_sanitizers)
endif()

# AddressSanitizer
if(USE_ASAN)
  set(NOVA_HAS_ASAN 1 CACHE INTERNAL "")

  if(MSVC)
    target_compile_options(nova_sanitizers INTERFACE /fsanitize=address)
  else()
    target_compile_options(nova_sanitizers INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(nova_sanitizers INTERFACE -fsanitize=address)
  endif()
endif()

# UndefinedBehaviorSanitizer
if(USE_UBSAN)
  if(MSVC)
    message(WARNING "UBSan is not supported on MSVC.")
  else()
    set(NOVA_HAS_UBSAN 1 CACHE INTERNAL "")
    target_compile_options(nova_sanitizers INTERFACE -fsanitize=undefined -fno-sanitize-recover=all)
    target_link_options(nova_sanitizers INTERFACE -fsanitize=undefined)
  endif()
endif()
