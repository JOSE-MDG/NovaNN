#[=======================================================================[.rst:
.. module:: NovaNNCPU
   :synopsis: CPU backend configuration for NovaNN targets.

Provides the ``nova_configure_cpu_target()`` function that applies all
detected SIMD flags, threading libraries, and OpenMP support to a
CMake target.

This module also includes the threading detection modules
(``DetectPThreads`` and ``DetectOpenMP``) and is included by
``NovaNNRuntime.cmake``.

.. function:: nova_configure_cpu_target(TARGET)

   Configure a target for the CPU backend.

   Applies the following to ``TARGET``:

   - All flags in ``SIMD_FLAGS`` as ``PRIVATE`` compile options.
   - Links ``Threads::Threads`` if pthreads is available.
   - Links ``OpenMP::OpenMP_C`` and ``OpenMP::OpenMP_CXX`` if OpenMP
     is found; defines ``NOVA_OPENMP=1`` on the target.
   - Defines ``NOVA_OPENMP=0`` if OpenMP is not found.

   :param TARGET: The target to configure (must already exist).
   :type TARGET:  ``target name``

   .. code-block:: cmake

      add_library(mylib OBJECT source.c)
      nova_configure_cpu_target(mylib)
#]=======================================================================]

include("${CMAKE_SOURCE_DIR}/cmake/Detect/threading/DetectPThreads.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/Detect/threading/DetectOpenMP.cmake")

function(nova_configure_cpu_target TARGET)
    if(SIMD_FLAGS)
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
