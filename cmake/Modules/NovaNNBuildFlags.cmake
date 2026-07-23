#[=======================================================================[.rst:
NovaNNBuildFlags
----------------

Centralized build flag configuration for the NovaNN project. Provides
functions to apply compiler warnings, debug/release optimization flags,
and linker hardening flags to targets. Includes ``DetectLTO`` and
``DetectSanitizers`` to resolve optimization dependencies.

.. note::
  NovaNN's MSVC support is much less optimized than for GCC/Clang (see
  ``CheckInstructionSupport.cmake``). This module still applies an
  equivalent-effort warning and hardening baseline to MSVC builds,
  mapped from the GCC/Clang flag set below. Where no direct MSVC
  equivalent exists, the flag is omitted and documented inline
  rather than silently dropped.

This module defines the following variables:

``NOVA_WARNING_FLAGS``
  List of compiler warning flags applied to all targets. GCC/Clang form
  shown; the ``MSVC`` branch inside each flag list carries the mapped
  ``/W4``-based equivalent via generator expressions.

``NOVA_CXX_FLAGS``
  C++-specific compiler flags (exceptions, RTTI, style warnings).

``NOVA_DEBUG_FLAGS``
  Flags applied in ``Debug`` configuration.

``NOVA_RELEASE_FLAGS``
  Flags applied in ``Release`` configuration.

This module defines the following functions:

.. command:: nova_configure_build_flags

  Apply compiler warnings, optimization flags, and sanitizer options
  to a target:

  .. code-block:: cmake

    nova_configure_build_flags(<target>)

.. command:: nova_configure_linker

  Apply linker hardening flags to a target:

  .. code-block:: cmake

    nova_configure_linker(<target>)

#]=======================================================================]

#[=======================================================================[.rst:
GCC/Clang -> MSVC warning flag mapping reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  -Wall -Wextra          -> /W4              (MSVC /Wall is too noisy in practice;
                                             /W4 is the practical maximum)
  -Wpedantic              -> /permissive-    (strict standard conformance)
  -Wshadow                -> /w14456 /w14457 /w14458 /w14459
  -Wcast-align            -> (none)          no direct MSVC warning; would need /analyze
  -Wconversion            -> /w14242 /w14254 /w14263
  -Wsign-conversion       -> /w14245 /w14365
  -Wfloat-equal           -> (none)          no direct MSVC warning
  -Wformat=2              -> /w14774 /w14777
  -Wformat-security       -> /w14774 /w14777 (same warnings cover both)
  -Wimplicit-fallthrough  -> /w15262
  -Wnull-dereference      -> (none)          requires /analyze (C6011); not a plain warning
  -Wpointer-arith         -> /w14826
  -Wundef                 -> (none)          no direct MSVC warning
  -Wuninitialized         -> /w14700
  -Wunused                -> covered by /W4
  -Wdouble-promotion      -> (none)          no direct MSVC warning
  -Wstrict-aliasing=2     -> (default)       MSVC always assumes strict aliasing
  -Wlogical-op            -> (none)          no direct MSVC warning
  -Wuseless-cast          -> (none)          no direct MSVC warning
  -Werror=return-type     -> /we4715         promote C4715 specifically to an error

  -fno-exceptions         -> /EHs-c-
  -fno-rtti               -> /GR-
  -Wold-style-cast        -> (partial)       no exact match; omitted
  -Wnon-virtual-dtor      -> /w14265
  -Woverloaded-virtual    -> /w14263
  -Wzero-as-null-pointer-constant -> /w14310 (approximate)
  -Wextra-semi            -> (none)          no direct MSVC warning
  -Wdeprecated            -> covered by /W4
  -Wregister              -> N/A            'register' keyword removed pre-C++17; no-op

  -O3                      -> /O2             MSVC has no /O3; /O2 is its ceiling
  -march=x86-64-v2/-mtune  -> (n/a)           MSVC has no microarch-level flag;
  -ffast-math              -> /fp:fast        different, less-standardized FP model
  -fstack-protector-strong -> /GS             enabled by default on MSVC
  -g                       -> /Zi
  -fno-omit-frame-pointer  -> /Oy-

  -Wl,-z,relro,-z,now      -> (n/a)          ELF-only concept, no PE equivalent
  -Wl,--as-needed          -> (n/a)          ELF-only concept
  -Wl,--no-undefined       -> (default)      link.exe already errors on unresolved
                                             symbols by default
  -Wl,-z,noexecstack       -> /NXCOMPAT      closest PE/DEP equivalent
                              + /DYNAMICBASE ASLR, PE hardening baseline
                              + /guard:cf    Control Flow Guard (COMPILER flag,
                                             not linker -- see note below)
                              + /GUARD:CF    Control Flow Guard (LINKER flag)

.. note::
  On CFG: /guard:cf (compiler) and /GUARD:CF (linker) are two distinct
  flags for the same feature and BOTH are required -- code compiled with
  /guard:cf but linked without /GUARD:CF pays the runtime check cost with
  no actual CFG protection in the resulting binary. This module therefore
  adds /guard:cf to NOVA_WARNING_FLAGS' MSVC compiler branch (not just the
  linker branch) so the two are never applied independently.

Reference: https://learn.microsoft.com/en-us/cpp/build/reference/compiler-options-listed-by-category
#]=======================================================================]

set(NOVA_WARNING_FLAGS
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
  -Wall -Wextra -Wpedantic
  -Wshadow -Wcast-align -Wconversion -Wfloat-equal
  -Wformat=2 -Wimplicit-fallthrough -Wnull-dereference
  -Wpointer-arith -Wsign-conversion -Wundef
  -Wuninitialized -Wunused -Wno-missing-field-initializers
  -Wno-unused-parameter -Wformat-security -Wdouble-promotion
  -Wstrict-aliasing=2
  -Werror=return-type -pipe
  >
  $<$<CXX_COMPILER_ID:GNU>:
  -Wlogical-op -Wuseless-cast
  >
  $<$<CXX_COMPILER_ID:MSVC>:
  /W4 /permissive-
  /w14456 /w14457 /w14458 /w14459
  /w14242 /w14254 /w14263
  /w14245 /w14365
  /w14774 /w14777
  /w15262
  /w14826
  /w14700
  /we4715
  /guard:cf
  >
)

set(NOVA_CXX_FLAGS
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
  -fno-exceptions -fno-rtti
  -Wpessimizing-move -Wredundant-move -Wold-style-cast
  -Wnon-virtual-dtor -Woverloaded-virtual -Wzero-as-null-pointer-constant
  -Wextra-semi -Wdeprecated -Wregister
  >
  $<$<CXX_COMPILER_ID:GNU>:
  -Wclass-memaccess -Wvolatile
  >
  $<$<CXX_COMPILER_ID:MSVC>:
  /EHs-c- /GR-
  /w14265 /w14263 /w14310
  >
)

set(NOVA_RELEASE_FLAGS
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
  -O3 -mtune=generic -march=x86-64-v2 -ffast-math -fno-finite-math-only
  -fstack-protector-strong
  $<$<AND:$<COMPILE_LANGUAGE:C>,$<C_COMPILER_ID:Clang>>:-fvectorize>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:-fvectorize>
  >
  $<$<CXX_COMPILER_ID:MSVC>:
  /O2 /fp:fast /GS
  >
)

set(NOVA_DEBUG_FLAGS
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-g -fno-omit-frame-pointer>
  $<$<CXX_COMPILER_ID:MSVC>:/Zi /Oy->
)

include(Detect/lto/DetectLTO)
include(Detect/sanitizers/DetectSanitizers)

#[=======================================================================[.rst:
.. command:: nova_configure_build_flags

  Apply compiler warnings and optimization flags to a target:

  .. code-block:: cmake

    nova_configure_build_flags(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function applies:

  - Warning flags from ``NOVA_WARNING_FLAGS`` (GCC/Clang or MSVC branch,
    selected automatically via ``CXX_COMPILER_ID`` generator expression).
  - C++-specific flags from ``NOVA_CXX_FLAGS`` (CXX language only).
  - Debug flags from ``NOVA_DEBUG_FLAGS`` in ``Debug`` configuration.
  - Release flags from ``NOVA_RELEASE_FLAGS`` in ``Release``
    configuration.
  - Sets ``INTERPROCEDURAL_OPTIMIZATION`` to ``ON`` and the
    ``-ffat-lto-objects`` flag when ``NOVA_HAS_LTO`` is true. LTO on
    MSVC is instead requested purely via ``INTERPROCEDURAL_OPTIMIZATION``
    (``/GL`` + ``/LTCG``, applied by CMake's MSVC LTO support);
    ``-ffat-lto-objects`` is a GCC-only flag and is skipped under MSVC.
  - Links against ``nova::sanitizers`` if AddressSanitizer or
    UndefinedBehaviorSanitizer are enabled.

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
    target_compile_options(${TARGET} PRIVATE
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-ffat-lto-objects>
    )
  endif()

  if(TARGET nova::sanitizers)
    target_link_libraries(${TARGET} PRIVATE nova::sanitizers)
  endif()
endfunction()

#[=======================================================================[.rst:
.. command:: nova_configure_linker

  Apply linker hardening flags to a target:

  .. code-block:: cmake

    nova_configure_linker(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function applies the following linker flags:

  On GCC/Clang (ELF/``ld``):

  - ``-Wl,-z,relro,-z,now`` for RELRO hardening.
  - ``-Wl,--as-needed`` to avoid unnecessary linking.
  - ``-Wl,--no-undefined`` to enforce symbol resolution.
  - ``-Wl,-z,noexecstack`` to mark the stack non-executable.

  On MSVC (PE/``link.exe``), these ELF-specific flags have no direct
  equivalent (see the mapping table at the top of this module). Instead
  the following PE hardening baseline is applied:

  - ``/DYNAMICBASE`` for ASLR.
  - ``/NXCOMPAT`` for DEP (closest analogue to ``noexecstack``).
  - ``/GUARD:CF`` for Control Flow Guard (linker-side; requires the
    matching ``/guard:cf`` compiler flag, applied in
    ``NOVA_WARNING_FLAGS``'s MSVC branch, to actually take effect).

  ``link.exe`` already fails on unresolved symbols by default, so no
  equivalent to ``--no-undefined`` is needed.

#]=======================================================================]
function(nova_configure_linker TARGET)
  target_link_options(${TARGET} PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
    -Wl,-z,relro,-z,now
    -Wl,--as-needed
    -Wl,--no-undefined
    -Wl,-z,noexecstack
    >
    $<$<CXX_COMPILER_ID:MSVC>:
    /DYNAMICBASE /NXCOMPAT /GUARD:CF
    >
  )
endfunction()
