#[=======================================================================[.rst:
NovaNNCPU
---------

Configure a CMake target for the CPU backend.  Applies SIMD compiler
flags, links pthreads if available, and configures OpenMP support.

This module includes ``DetectPThreads`` and ``DetectOpenMP`` to
resolve threading dependencies.

This module defines the following functions:

.. command:: nova_configure_cpu_target

  Configure a target with CPU-specific compile options:

  .. code-block:: cmake

    nova_configure_cpu_target(<target>)

#]=======================================================================]

include(Detect/threading/DetectPThreads)
include(Detect/threading/DetectOpenMP)

#[=======================================================================[.rst:
.. command:: nova_configure_cpu_target

  Configure a target with CPU-specific compile options:

  .. code-block:: cmake

    nova_configure_cpu_target(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function performs the following actions:

  - Appends ``SIMD_FLAGS`` to the target compile options.
  - Links ``Threads::Threads`` when pthreads is available.
  - Links ``OpenMP::OpenMP_C`` and ``OpenMP::OpenMP_CXX`` when OpenMP
    is available, and defines ``NOVA_OPENMP=1``.
  - Defines ``NOVA_OPENMP=0`` when OpenMP is absent.

#]=======================================================================]
function(nova_configure_cpu_target TARGET)
    # SIMD_FLAGS is populated by check_simd() with -mavx512f / -mfma / etc.
    # (GNU-style flags). MSVC never runs that detection path: NovaNN's
    # SIMD internals use [[{gnu,clang}::target(...)]], which cl.exe does not support,
    # so MSVC builds always fall back to the scalar kernel implementations
    # regardless of what the host CPU actually supports. Applying these
    # flags under MSVC would either be silently ignored or misinterpreted.
    if(SIMD_FLAGS AND NOT MSVC)
        target_compile_options(${TARGET} PRIVATE ${SIMD_FLAGS})
    endif()

    if(NOVA_HAS_PTHREADS)
        target_link_libraries(${TARGET} PRIVATE Threads::Threads)
    endif()

    if(NOVA_HAS_OPENMP)
        if(TARGET OpenMP::OpenMP_C)
            target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_C)
        endif()

        if(TARGET OpenMP::OpenMP_CXX)
            target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
        endif()

        target_compile_definitions(${TARGET} PRIVATE NOVA_OPENMP=1)
    else()
        target_compile_definitions(${TARGET} PRIVATE NOVA_OPENMP=0)
    endif()
endfunction()
