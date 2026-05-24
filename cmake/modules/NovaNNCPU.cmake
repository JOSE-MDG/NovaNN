#[[
    @file NovaNNCPU.cmake
    @brief CPU back-end target configuration.

    Provides the public function:

        novaNN_configure_cpu_target(<target>)

    which applies three layers to the given CMake target:
      1. SIMD_FLAGS       — compiler flags for detected SIMD extensions
      2. pthreads          — links Threads::Threads if NOVA_HAS_PTHREADS
      3. OpenMP            — links OpenMP::OpenMP_C / OpenMP::OpenMP_CXX
                            per compile language; defines NOVA_OPENMP=1 or 0

    Detects threading back-ends (pthreads, OpenMP) on every call if not
    already cached.  Both detectors have multiple-inclusion guards.

    Usage:
        novaNN_configure_cpu_target(my_target)

    See also : DetectSIMD.cmake   — populates SIMD_FLAGS and HAS_* variables
               DetectPThreads.cmake — sets NOVA_HAS_PTHREADS
               DetectOpenMP.cmake  — sets NOVA_HAS_OPENMP
]]
include("${CMAKE_SOURCE_DIR}/cmake/detect/threading/DetectPThreads.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/detect/threading/DetectOpenMP.cmake")

function(novaNN_configure_cpu_target TARGET)
    # SIMD flags (populated by DetectSIMD)
    target_compile_options(${TARGET} PRIVATE ${SIMD_FLAGS})

    # pthreads
    if(NOVA_HAS_PTHREADS)
        target_link_libraries(${TARGET} PRIVATE Threads::Threads)
    endif()

    # OpenMP — C kernels and C++ autograd separately
    if(NOVA_HAS_OPENMP)
        target_link_libraries(${TARGET} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:OpenMP::OpenMP_C>
            $<$<COMPILE_LANGUAGE:CXX>:OpenMP::OpenMP_CXX>
        )
        target_compile_definitions(${TARGET} PRIVATE NOVA_OPENMP=1)
    else()
        target_compile_definitions(${TARGET} PRIVATE NOVA_OPENMP=0)
    endif()
endfunction()
