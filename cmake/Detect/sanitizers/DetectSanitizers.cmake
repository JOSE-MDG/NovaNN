#[=======================================================================[.rst:
.. module:: DetectSanitizers
   :synopsis: Runtime sanitizer configuration for NovaNN.

Configures compiler and linker options for runtime sanitizers.
Sanitizers are debugging tools that detect memory errors and undefined
behavior at runtime.  They are disabled by default and intended for
development and testing use.

**Supported sanitizers:**

- **AddressSanitizer (ASan)** — Detects use-after-free, buffer
  overflows, and memory leaks.  Compatible with Valgrind (not
  simultaneously).
- **UndefinedBehaviorSanitizer (UBSan)** — Detects undefined behavior
  such as integer overflow, null pointer dereference, and invalid
  shifts.

**Options consumed:**

- ``USE_ASAN`` — Enable AddressSanitizer (default: ``OFF``).
- ``USE_UBSAN`` — Enable UndefinedBehaviorSanitizer (default: ``OFF``).

**Output variables:**

- ``NOVA_HAS_ASAN`` — ``1`` if ASan is enabled, ``0`` otherwise.
- ``NOVA_HAS_UBSAN`` — ``1`` if UBSan is enabled, ``0`` otherwise.

.. note::

   This module is included by ``NovaNNBuildFlags.cmake`` and does not
   need to be included directly.

.. warning::

   Do not enable multiple sanitizers that are incompatible with each
   other.  ASan and TSan are mutually exclusive.  ASan and UBSan can
   be combined.
#]=======================================================================]

if(NOT USE_ASAN AND NOT USE_UBSAN)
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
