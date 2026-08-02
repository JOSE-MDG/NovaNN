#[=======================================================================[.rst:
NovaNNBuildFlags
----------------

Centralized build flag configuration for the NovaNN project. Provides
functions to apply compiler warnings, debug/release optimization flags,
and linker hardening flags to targets. Includes ``DetectLTO`` and
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

  Apply compiler warnings, optimization flags, and sanitizer options
  to a target:

  .. code-block:: cmake

    nova_configure_build_flags(<target>)

.. command:: nova_configure_linker

  Apply linker hardening flags to a target:

  .. code-block:: cmake

    nova_configure_linker(<target>)

#]=======================================================================]

# Detect clang-cl (Clang with MSVC frontend)
set(_nova_is_clang_cl FALSE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
   CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(_nova_is_clang_cl TRUE)
endif()

# Helper: emit a flag as /clang:<flag> on clang-cl, plain otherwise.
macro(_nova_wrap_flag _out _flag)
  if(_nova_is_clang_cl)
    set(${_out} "/clang:${_flag}")
  else()
    set(${_out} "${_flag}")
  endif()
endmacro()


# Warning flags
if(_nova_is_clang_cl)
  # clang-cl accepts /W4 for warnings; Clang -W flags also work when prefixed
  # with /clang:, but many are redundant or cause issues on the MSVC frontend.
  # We use the /W4 + explicit additions approach.
  set(NOVA_WARNING_FLAGS
    /W4
    /WX-                          # treat warnings as non-fatal globally;
                                  # use /WX per-target if desired
    /wd4100                       # unreferenced formal parameter (like -Wno-unused-parameter)
    /wd4201                       # nonstandard extension: nameless struct/union
    /wd4244                       # conversion, possible loss of data (too noisy)
    /wd4267                       # size_t -> int conversion (too noisy)
    /wd4456 /wd4457               # declaration hides local / parameter
    /clang:-Wshadow
    /clang:-Wcast-align
    /clang:-Wformat=2
    /clang:-Wimplicit-fallthrough
    /clang:-Wnull-dereference
    /clang:-Wpointer-arith
    /clang:-Wundef
    /clang:-Wuninitialized
    /clang:-Wdouble-promotion
    /clang:-Wstrict-aliasing=2
    /clang:-Werror=return-type
    /guard:cf
  )
else()
  set(NOVA_WARNING_FLAGS
    -Wall -Wextra -Wpedantic
    -Wshadow -Wcast-align -Wconversion -Wfloat-equal
    -Wformat=2 -Wimplicit-fallthrough -Wnull-dereference
    -Wpointer-arith -Wsign-conversion -Wundef
    -Wuninitialized -Wunused -Wno-missing-field-initializers
    -Wno-unused-parameter -Wformat-security -Wdouble-promotion
    -Wstrict-aliasing=2
    -Werror=return-type -pipe
  )

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND NOVA_WARNING_FLAGS -Wlogical-op -Wuseless-cast)
  endif()

  if(WIN32)
    list(APPEND NOVA_WARNING_FLAGS /guard:cf)
  endif()
endif()


# C++-specific flags
if(_nova_is_clang_cl)
  set(NOVA_CXX_FLAGS
    /clang:-fno-exceptions
    /clang:-fno-rtti
    /clang:-Wpessimizing-move
    /clang:-Wredundant-move
    /clang:-Wnon-virtual-dtor
    /clang:-Woverloaded-virtual
    /clang:-Wzero-as-null-pointer-constant
    /clang:-Wextra-semi
    /clang:-Wdeprecated
  )
else()
  set(NOVA_CXX_FLAGS
    -fno-exceptions -fno-rtti
    -Wpessimizing-move -Wredundant-move -Wold-style-cast
    -Wnon-virtual-dtor -Woverloaded-virtual -Wzero-as-null-pointer-constant
    -Wextra-semi -Wdeprecated -Wregister
  )

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND NOVA_CXX_FLAGS -Wclass-memaccess -Wvolatile)
  endif()
endif()


# Release flags
if(_nova_is_clang_cl)
  set(NOVA_RELEASE_FLAGS
    /O2
    /clang:-ffast-math
    /clang:-fno-finite-math-only
    /GS            # buffer security check (clang-cl equivalent of -fstack-protector-strong)
  )
else()
  set(NOVA_RELEASE_FLAGS
    -O3 -ffast-math -fno-finite-math-only
    -fstack-protector-strong
  )

  if(NOT WIN32)
    list(APPEND NOVA_RELEASE_FLAGS -mtune=generic -march=x86-64-v2)
  endif()
endif()


# Debug flags
if(_nova_is_clang_cl)
  set(NOVA_DEBUG_FLAGS
    /Zi                           # debug info (PDB)
    /clang:-fno-omit-frame-pointer
  )
else()
  set(NOVA_DEBUG_FLAGS
    -g -fno-omit-frame-pointer
  )
endif()

include(Detect/lto/DetectLTO)
include(Detect/sanitizers/DetectSanitizers)

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
  - Sets ``INTERPROCEDURAL_OPTIMIZATION`` to ``ON`` and the
    ``-ffat-lto-objects`` flag when ``NOVA_HAS_LTO`` is true.
    ``-ffat-lto-objects`` is a GCC-only flag and is skipped on Clang.
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

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(_nova_is_clang_cl)
      target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:C>:/clang:-fvectorize>
        $<$<COMPILE_LANGUAGE:CXX>:/clang:-fvectorize>
      )
    else()
      target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-fvectorize>
        $<$<COMPILE_LANGUAGE:CXX>:-fvectorize>
      )
    endif()
  endif()

  if(NOVA_HAS_LTO)
    set_target_properties(${TARGET} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(${TARGET} PRIVATE -ffat-lto-objects)
    endif()
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

  On Linux (ELF), this function applies:

  - ``-Wl,-z,relro,-z,now`` for RELRO hardening.
  - ``-Wl,--as-needed`` to avoid unnecessary linking.
  - ``-Wl,--no-undefined`` to enforce symbol resolution.
  - ``-Wl,-z,noexecstack`` to mark the stack non-executable.

  On Windows (PE) with clang-cl, an equivalent hardening baseline
  is applied via:

  - ``/DYNAMICBASE`` for ASLR.
  - ``/NXCOMPAT`` for DEP (closest analogue to ``noexecstack``).
  - ``/GUARD:CF`` for Control Flow Guard (linker-side; requires the
    matching ``/guard:cf`` compiler flag in ``NOVA_WARNING_FLAGS``
    to actually take effect).

#]=======================================================================]
function(nova_configure_linker TARGET)
  if(WIN32)
    target_link_options(${TARGET} PRIVATE
      /DYNAMICBASE /NXCOMPAT /GUARD:CF
    )
  else()
    target_link_options(${TARGET} PRIVATE
      -Wl,-z,relro,-z,now
      -Wl,--as-needed
      -Wl,--no-undefined
      -Wl,-z,noexecstack
    )
  endif()
endfunction()
