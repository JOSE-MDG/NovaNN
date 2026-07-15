#[=======================================================================[.rst:
DetectSanitizers
----------------

Configure AddressSanitizer (ASan) and UndefinedBehaviorSanitizer
(UBSan) compile and link options.  The module consults the user
options ``USE_ASAN`` and ``USE_UBSAN``.

If neither option is enabled the module returns immediately.

This module sets the following cache variables:

``NOVA_HAS_ASAN``
  ``1`` when ``USE_ASAN`` is ``ON``.

``NOVA_HAS_UBSAN``
  ``1`` when ``USE_UBSAN`` is ``ON``.

When enabled the module adds global compile and link options via
:command:`add_compile_options` and :command:`add_link_options`.

#]=======================================================================]

if(NOT USE_ASAN AND NOT USE_UBSAN)
  return()
endif()

if(MSVC)
  return()
endif()

if(USE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
  set(NOVA_HAS_ASAN 1 CACHE INTERNAL "")
endif()

if(USE_UBSAN)
  add_compile_options(-fsanitize=undefined -fno-sanitize-recover=all)
  add_link_options(-fsanitize=undefined)
  set(NOVA_HAS_UBSAN 1 CACHE INTERNAL "")
endif()
