#[[
    @file NovaNNCPU.cmake
    @brief CPU back-end target configuration.

    Provides the public function:

        novaNN_configure_cpu_target(<target>)

    which applies three layers to the given CMake target:
      1. SIMD_FLAGS       — compiler flags for detected SIMD extensions
      2. pthreads          — links Threads::Threads if NOVA_HAS_PTHREADS
      3. OpenMP            — links the C and/or C++ OpenMP target required by
                            the target sources; defines NOVA_OPENMP=1 or 0

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

    # OpenMP
    if(NOVA_HAS_OPENMP)
        get_target_property(_target_sources ${TARGET} SOURCES)
        set(_has_c OFF)
        set(_has_cxx OFF)

        foreach(_source IN LISTS _target_sources)
            if(_source MATCHES "\\.c$")
                set(_has_c ON)
            elseif(_source MATCHES "\\.(cc|cpp|cxx|c\\+\\+)$")
                set(_has_cxx ON)
            endif()
        endforeach()

        if(_has_c)
            target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_C)
        endif()

        if(_has_cxx OR (NOT _has_c AND NOT _has_cxx))
            target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
        endif()

        target_compile_definitions(${TARGET} PRIVATE NOVA_OPENMP=1)
    else()
        target_compile_definitions(${TARGET} PRIVATE NOVA_OPENMP=0)
    endif()
endfunction()
