#[=======================================================================[.rst:
NovaNNCPU
---------

Configure a CMake target for the CPU backend: SIMD compiler flags,
pthreads and OpenMP.  Includes ``DetectPThreads`` and ``DetectOpenMP``
to resolve threading dependencies.
#]=======================================================================]

include(Detect/threading/DetectPThreads)
include(Detect/threading/DetectOpenMP)

#[=======================================================================[.rst:
.. command:: nova_configure_cpu_target

  Configure a target with CPU-specific compile options:

  .. code-block:: cmake

    nova_configure_cpu_target(<target>)

#]=======================================================================]
function(nova_configure_cpu_target TARGET)
    if(SIMD_FLAGS)
        target_compile_options(${TARGET} PRIVATE ${SIMD_FLAGS})
    endif()

    if(NOVA_HAS_PTHREADS)
        target_link_libraries(${TARGET} PRIVATE Threads::Threads)
    endif()

    if(NOVA_HAS_OPENMP)
        if(TARGET nova::openmp)
            target_link_libraries(${TARGET} PRIVATE nova::openmp)
        endif()

        target_compile_definitions(${TARGET} PRIVATE NOVA_OPENMP=1)
    endif()
endfunction()
