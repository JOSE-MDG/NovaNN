#[=======================================================================[.rst:
.. module:: NovaNNBuildFlags
   :synopsis: Centralized compiler and linker flag management for NovaNN.

Defines all compiler warning flags, C++-specific flags, and
configuration-dependent optimization flags.  Provides two functions
consumed by ``CMakeLists.txt`` files to apply these flags to targets.

This module also includes the LTO and sanitizer detection modules.

**Variables defined:**

- ``NOVA_WARNING_FLAGS`` — Compiler warning flags applied to all
  languages.
- ``NOVA_CXX_FLAGS`` — C++-only flags (applied via generator
  expression).
- ``NOVA_DEBUG_FLAGS`` — Debug configuration optimization flags.
- ``NOVA_RELEASE_FLAGS`` — Release configuration optimization flags.

**Functions provided:**

- ``nova_configure_build_flags(TARGET)`` — Applies all compile options
  (warnings, CXX flags, config-dependent optimizations) and LTO to
  the given target.
- ``nova_configure_linker(TARGET)`` — Applies linker hardening flags
  (RELRO, as-needed, no-undefined) to the given target.

**Included detection modules:**

- ``DetectLTO.cmake`` — Link-Time Optimization detection.
- ``DetectSanitizers.cmake`` — ASan/UBSan configuration.

.. note::

   This module is included by ``NovaNNRuntime.cmake`` and does not
   need to be included directly.

.. code-block:: cmake

   include(Modules/NovaNNBuildFlags)
   nova_configure_build_flags(my_target)
   nova_configure_linker(my_target)
#]=======================================================================]

set(NOVA_WARNING_FLAGS
    -Wall -Werror -Wextra -Wpedantic
    -Wshadow -Wcast-align -Wconversion -Wfloat-equal
    -Wformat=2 -Wimplicit-fallthrough -Wnull-dereference
    -Wpointer-arith -Wsign-conversion -Wundef
    -Wuninitialized -Wunused
    -Wno-missing-field-initializers -Wno-unused-parameter
    -pipe
)

set(NOVA_CXX_FLAGS
    -fno-exceptions -fno-rtti
    -Wpessimizing-move -Wredundant-move -Wold-style-cast
)

set(NOVA_DEBUG_FLAGS -g -fno-omit-frame-pointer)
set(NOVA_RELEASE_FLAGS -O3 -march=native -ffast-math)

include("${CMAKE_SOURCE_DIR}/cmake/Detect/lto/DetectLTO.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Detect/sanitizers/DetectSanitizers.cmake")

function(nova_configure_build_flags TARGET)
    target_compile_options(${TARGET} PRIVATE
        ${NOVA_WARNING_FLAGS}
        $<$<COMPILE_LANGUAGE:CXX>:${NOVA_CXX_FLAGS}>
        $<$<CONFIG:Debug>:${NOVA_DEBUG_FLAGS}>
        $<$<CONFIG:Release>:${NOVA_RELEASE_FLAGS}>
    )

    if(NOVA_HAS_LTO)
        set_target_properties(${TARGET} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endfunction()

function(nova_configure_linker TARGET)
    target_link_options(${TARGET} PRIVATE
        -Wl,-z,relro,-z,now
        -Wl,--as-needed
        -Wl,--no-undefined
    )
endfunction()
