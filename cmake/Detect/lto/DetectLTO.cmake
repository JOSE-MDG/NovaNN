#[=======================================================================[.rst:
.. module:: DetectLTO
   :synopsis: Link-Time Optimization detection for NovaNN.

Detects compiler support for Link-Time Optimization (LTO) and sets
the ``CMAKE_INTERPROCEDURAL_OPTIMIZATION`` variable when supported.

LTO enables the compiler to optimize across translation unit boundaries
during the link step, allowing cross-file inlining and dead code
elimination.  This is particularly beneficial for NovaNN because the
SIMD dispatch functions in ``cast.h`` are ``inline`` and called from
multiple translation units.

**Options consumed:**

- ``USE_LTO`` — Enable LTO detection (default: ``ON``).

**Output variables:**

- ``NOVA_HAS_LTO`` — ``1`` if LTO is supported and enabled, ``0``
  otherwise.

.. note::

   This module is included by ``NovaNNBuildFlags.cmake`` and does not
   need to be included directly.
#]=======================================================================]

if(NOVA_HAS_LTO)
    return()
endif()

if(NOT USE_LTO)
    set(NOVA_HAS_LTO 0 CACHE INTERNAL "")
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
