#[=======================================================================[.rst:
NovaNNBuildFlags
----------------

Centralized build flag configuration for the NovaNN project.  Provides
functions to apply compiler warnings, debug/release optimization flags,
and linker hardening flags to targets.  Includes ``DetectLTO`` and
``DetectSanitizers`` to resolve optimization dependencies.

This module defines the following variables:

``NOVA_WARNING_FLAGS``
  List of compiler warning flags applied to all targets.

``NOVA_CXX_FLAGS``
  C++-specific compiler flags (exceptions, RTTI, style warnings).

``NOVA_DEBUG_FLAGS``
  Flags applied in ``Debug`` configuration.

``NOVA_RELEASE_FLAGS``
  Flags applied in ``Release`` configuration.

This module defines the following functions:

.. command:: nova_configure_build_flags

  Apply compiler warnings and optimization flags to a target:

  .. code-block:: cmake

    nova_configure_build_flags(<target>)

.. command:: nova_configure_linker

  Apply linker hardening flags to a target:

  .. code-block:: cmake

    nova_configure_linker(<target>)

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

set(NOVA_RELEASE_FLAGS
  -O3 -march=native -ffast-math -fno-finite-math-only
  $<$<AND:$<COMPILE_LANGUAGE:C>,$<C_COMPILER_ID:Clang>>:-fvectorize>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:-fvectorize>
)
set(NOVA_DEBUG_FLAGS -g -fno-omit-frame-pointer)

include("${CMAKE_SOURCE_DIR}/cmake/Detect/lto/DetectLTO.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Detect/sanitizers/DetectSanitizers.cmake")

#[=======================================================================[.rst:
.. command:: nova_configure_build_flags

  Apply compiler warnings and optimization flags to a target:

  .. code-block:: cmake

    nova_configure_build_flags(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function applies:

  - Warning flags from ``NOVA_WARNING_FLAGS``.
  - C++-specific flags from ``NOVA_CXX_FLAGS`` (CXX language only).
  - Debug flags from ``NOVA_DEBUG_FLAGS`` in ``Debug`` configuration.
  - Release flags from ``NOVA_RELEASE_FLAGS`` in ``Release``
    configuration.
  - Sets ``INTERPROCEDURAL_OPTIMIZATION`` to ``ON`` when
    ``NOVA_HAS_LTO`` is true.

#]=======================================================================]
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

#[=======================================================================[.rst:
.. command:: nova_configure_linker

  Apply linker hardening flags to a target:

  .. code-block:: cmake

    nova_configure_linker(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function applies the following linker flags:

  - ``-Wl,-z,relro,-z,now`` for RELRO hardening.
  - ``-Wl,--as-needed`` to avoid unnecessary linking.
  - ``-Wl,--no-undefined`` to enforce symbol resolution.

#]=======================================================================]
function(nova_configure_linker TARGET)
  target_link_options(${TARGET} PRIVATE
    -Wl,-z,relro,-z,now
    -Wl,--as-needed
    -Wl,--no-undefined
  )
endfunction()
