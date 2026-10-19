#[=======================================================================[.rst:
DetectLTO
---------

Detect Link-Time Optimization (LTO / IPO) support.  When the user
option ``USE_LTO`` is enabled the module uses
:command:`check_ipo_supported` to verify compiler and linker support
and sets ``CMAKE_INTERPROCEDURAL_OPTIMIZATION`` to ``ON`` globally.

``NOVA_HAS_LTO``
  ``1`` if LTO is supported and enabled, ``0`` otherwise.

The module is idempotent: it returns early when ``NOVA_HAS_LTO`` is
already set, and sets it to ``0`` without probing when ``USE_LTO`` is
``OFF``.

#]=======================================================================]

if(NOT USE_LTO)
    set(NOVA_HAS_LTO 0 CACHE INTERNAL "")
    return()
endif()

if(NOVA_HAS_LTO)
    return()
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT _nova_ipo_supported OUTPUT _nova_ipo_error)

if(_nova_ipo_supported)
    set(NOVA_HAS_LTO 1 CACHE INTERNAL "")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
else()
    set(NOVA_HAS_LTO 0 CACHE INTERNAL "")
    message(WARNING "LTO not supported: ${_nova_ipo_error}")
endif()
