#[=======================================================================[.rst:
NovaNNBuildFlags
----------------

Centralized host build flag configuration for the NovaNN project.
Provides functions to apply compiler warnings, per-configuration
optimization flags, and linker hardening flags to targets.  The flags
apply to C and C++ only; CUDA and HIP device code is configured by the
backend modules.  Includes ``DetectLTO`` and ``DetectSanitizers``.

This module defines the following variables:

``NOVA_COMMON_FLAGS``
  Flags applied to every C and C++ translation unit.

``NOVA_WARNING_FLAGS``
  List of compiler warning flags applied to all targets.

``NOVA_CXX_FLAGS``
  C++-specific compiler flags (exceptions, RTTI, style warnings).

``NOVA_DEBUG_FLAGS``
  Flags applied in ``Debug`` configuration.

``NOVA_RELEASE_FLAGS``
  Flags applied in ``Release`` configuration.

``NOVA_RELWITHDEBINFO_FLAGS``
  Flags applied in ``RelWithDebInfo`` configuration.

``NOVA_MINSIZEREL_FLAGS``
  Flags applied in ``MinSizeRel`` configuration.

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

# The Windows SDK defines min and max as macros (minwindef.h) and the
# reduced-precision dtypes declare min() and max() members, so the
# declaration itself fails without NOMINMAX.
if(WIN32)
  add_compile_definitions(NOMINMAX)
endif()


if(_nova_is_clang_cl)
  set(NOVA_COMMON_FLAGS
    /clang:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.
    /clang:-ffile-prefix-map=${CMAKE_BINARY_DIR}=build
    /clang:-fstrict-flex-arrays=3
  )
else()
  set(NOVA_COMMON_FLAGS
    -ffile-prefix-map=${CMAKE_SOURCE_DIR}=.
    -ffile-prefix-map=${CMAKE_BINARY_DIR}=build
    -fstrict-flex-arrays=3
  )
endif()


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
  # No -fno-rtti here: the MSVC-like frontend turns off static RTTI with
  # it, and the MSVC standard library then ships <any> and <variant>
  # empty (STL4040).  GoogleTest needs std::any.
  set(NOVA_CXX_FLAGS
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
    -fno-rtti
    -fvisibility-inlines-hidden
    -Wpessimizing-move -Wredundant-move -Wold-style-cast
    -Wnon-virtual-dtor -Woverloaded-virtual -Wzero-as-null-pointer-constant
    -Wextra-semi -Wdeprecated -Wregister
  )

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND NOVA_CXX_FLAGS -Wclass-memaccess -Wvolatile)
  endif()
endif()


# Release flags
#
# -ffast-math is a group flag: the overrides after it restore the IEEE
# guarantees the library depends on.
if(_nova_is_clang_cl)
  set(_nova_fast_math_flags
    /clang:-ffast-math
    /clang:-fno-finite-math-only
    /clang:-fsigned-zeros
    /clang:-fno-associative-math
    /clang:-fno-reciprocal-math
    /clang:-ffp-contract=fast
  )

  set(NOVA_RELEASE_FLAGS
    /GS
    /Gy
    /Gw
    /Zc:inline
    ${_nova_fast_math_flags}
  )
  list(APPEND NOVA_RELEASE_FLAGS /clang:-march=x86-64-v2)

  set(NOVA_RELWITHDEBINFO_FLAGS
    /GS
    /Gy
    /Gw
    /Zc:inline
    ${_nova_fast_math_flags}
  )
  list(APPEND NOVA_RELWITHDEBINFO_FLAGS /clang:-march=x86-64-v2)

  set(NOVA_MINSIZEREL_FLAGS
    /GS
    /Gy
    /Gw
    /Zc:inline
    /clang:-ffp-contract=fast
    /clang:-march=x86-64-v2
  )
else()
  set(_nova_fast_math_flags
    -ffast-math
    -fno-finite-math-only
    -fsigned-zeros
    -fno-associative-math
    -fno-reciprocal-math
    -ffp-contract=fast
  )

  # -fcf-protection and -mharden-sls are x86 only.
  set(_nova_hardening_flags)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    list(APPEND _nova_hardening_flags
      -fstack-clash-protection
      -fcf-protection=full
      -mharden-sls=all
    )
  endif()
  list(APPEND _nova_hardening_flags
    -D_FORTIFY_SOURCE=3
  )

  # Safe because nova is a DSO linked with -z now.
  set(_nova_perf_flags
    -fno-semantic-interposition
    -fno-plt
  )

  set(NOVA_RELEASE_FLAGS
    -O3
    ${_nova_fast_math_flags}
    -fstack-protector-strong
    ${_nova_hardening_flags}
    ${_nova_perf_flags}
    -mtune=generic -march=x86-64-v2
  )

  set(NOVA_RELWITHDEBINFO_FLAGS
    ${_nova_fast_math_flags}
    -fstack-protector-strong
    ${_nova_hardening_flags}
    ${_nova_perf_flags}
    -mtune=generic -march=x86-64-v2
  )

  # No -fno-semantic-interposition: it can grow code.
  set(NOVA_MINSIZEREL_FLAGS
    -ffp-contract=fast
    -fstack-protector-strong
    ${_nova_hardening_flags}
    -fno-plt
    -mtune=generic -march=x86-64-v2
  )
endif()


# -ffp-contract=off keeps Debug reproducible; -Og keeps it debuggable.
if(_nova_is_clang_cl)
  set(NOVA_DEBUG_FLAGS
    /Z7
    /clang:-Og
    /clang:-fno-omit-frame-pointer
    /clang:-ffp-contract=off
  )
else()
  set(NOVA_DEBUG_FLAGS
    -Og
    -g3
    -gdwarf-5
    -fno-omit-frame-pointer
    -D_GLIBCXX_ASSERTIONS
    -ffp-contract=off
  )
endif()

include(Detect/lto/DetectLTO)
include(Detect/sanitizers/DetectSanitizers)

#[=======================================================================[.rst:
.. command:: nova_configure_build_flags

  Apply compiler warnings and optimization flags to a target:

  .. code-block:: cmake

    nova_configure_build_flags(<target>)

  Apply the ``NOVA_*_FLAGS`` lists per configuration to C and C++.
  Enable ``INTERPROCEDURAL_OPTIMIZATION`` under ``NOVA_HAS_LTO``
  (``-ffat-lto-objects`` on GCC only) and link ``nova::sanitizers``.

#]=======================================================================]
function(nova_configure_build_flags TARGET)
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:C,CXX>:${NOVA_COMMON_FLAGS}>
    $<$<COMPILE_LANGUAGE:C,CXX>:${NOVA_WARNING_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${NOVA_CXX_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Debug>>:${NOVA_DEBUG_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Release>>:${NOVA_RELEASE_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:RelWithDebInfo>>:${NOVA_RELWITHDEBINFO_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:MinSizeRel>>:${NOVA_MINSIZEREL_FLAGS}>
  )

  if(NOT WIN32)
    # The driver links crtfastmath.o under -ffast-math, which sets FTZ/DAZ
    # process-wide and flushes subnormals. This must be visible at LINK time
    # (the decision happens during linking), not just at compile time.
    target_link_options(${TARGET} PRIVATE -mno-daz-ftz)
  endif()

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
      target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-ffat-lto-objects>
      )
    endif()
  endif()

  if(TARGET nova::sanitizers)
    target_link_libraries(${TARGET} PRIVATE nova::sanitizers)
  endif()

  # On Windows with ASan, deploy the ASan runtime DLL alongside executables
  # and shared libraries (required at load time for clang-cl dynamic runtime).
  if(WIN32 AND NOVA_HAS_ASAN AND DEFINED NOVA_ASAN_RUNTIME_DLL)
    nova_deploy_asan_runtime(${TARGET})
  endif()

  # Inject the UBSan runtime into shared libraries only. Executables must not
  # inject it: the runtime's constructor then runs inside ld.so at startup
  # and deadlocks re-entering its own sigaction interceptor before main.
  # Executables get the runtime from the compiler driver via -fsanitize.
  # Static and object libraries must not link it either: CMake propagates
  # PRIVATE dependencies of static libraries to consumers ($<LINK_ONLY:...>),
  # which would smuggle the runtime back into every executable downstream.
  if(TARGET nova::ubsan_runtime)
    get_target_property(_nova_san_target_type ${TARGET} TYPE)
    if(_nova_san_target_type STREQUAL "SHARED_LIBRARY")
      target_link_libraries(${TARGET} PRIVATE nova::ubsan_runtime)
    endif()
  endif()
endfunction()

#[=======================================================================[.rst:
.. command:: nova_configure_linker

  Apply linker hardening flags to a target:

  .. code-block:: cmake

    nova_configure_linker(<target>)

  On Linux (ELF): RELRO, ``-Wl,--as-needed``, ``-Wl,--no-undefined``
  and a non-executable stack.

  On Windows (PE) with clang-cl, the equivalent hardening baseline:
  ``/DYNAMICBASE`` (ASLR), ``/NXCOMPAT`` (DEP) and ``/GUARD:CF``
  (Control Flow Guard on the linker side; it requires the matching
  ``/guard:cf`` compiler flag in ``NOVA_WARNING_FLAGS`` to actually
  take effect).  Optimized configurations also get ``/OPT:REF``,
  ``/OPT:ICF=4`` and ``/Brepro``.

#]=======================================================================]
function(nova_configure_linker TARGET)
  if(WIN32)
    target_link_options(${TARGET} PRIVATE
      /DYNAMICBASE /NXCOMPAT /GUARD:CF
      /Brepro
    )
    target_link_options(${TARGET} PRIVATE
      $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:/OPT:REF>
      $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:/OPT:ICF=4>
      $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:/DEBUG:FULL>
      $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:/PDBALTPATH:%_PDB%>
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
