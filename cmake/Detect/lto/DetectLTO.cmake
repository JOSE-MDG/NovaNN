#[=======================================================================[.rst:
DetectLTO
---------

Detect Link-Time Optimization (LTO / IPO) support.  When the user
option ``USE_LTO`` is enabled the module uses
:command:`check_ipo_supported` to verify compiler and linker support.

This module sets the following cache variables:

``NOVA_HAS_LTO``
  ``1`` if LTO is supported and enabled, ``0`` otherwise.

When LTO is available the module also sets
``CMAKE_INTERPROCEDURAL_OPTIMIZATION`` to ``ON`` globally.

The module is idempotent: it skips the detection when ``NOVA_HAS_LTO``
already matches the requested ``USE_LTO`` state.  If ``USE_LTO`` is
``OFF`` the module sets ``NOVA_HAS_LTO`` to ``0`` and returns without
performing a detection check.

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
