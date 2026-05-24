#[[
    @file DetectOpenMP.cmake
    @brief OpenMP detection for C and C++.

    Uses `find_package(OpenMP COMPONENTS C CXX)` to detect OpenMP support for
    both languages.  On success (both OpenMP_C_FOUND and OpenMP_CXX_FOUND):
      - Sets NOVA_HAS_OPENMP = 1
      - Exposes OpenMP::OpenMP_C and OpenMP::OpenMP_CXX imported targets
    Otherwise:
      - Sets NOVA_HAS_OPENMP = 0

    Multiple-inclusion guard: returns early if NOVA_HAS_OPENMP is already
    defined, so it is safe to include from multiple call sites.

    Variables set:
      - NOVA_HAS_OPENMP  — 1 if OpenMP is available for both C and CXX, 0 otherwise

    Targets exposed (conditional):
      - OpenMP::OpenMP_C   — OpenMP C compiler support
      - OpenMP::OpenMP_CXX — OpenMP CXX compiler support

    Usage:
        include(detect/threading/DetectOpenMP)

    See also : NovaNNCPU.cmake — consumes NOVA_HAS_OPENMP for target config
]]
if(DEFINED NOVA_HAS_OPENMP)
    return()
endif()

find_package(OpenMP COMPONENTS C CXX)

if(OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
    set(NOVA_HAS_OPENMP 1)
    message(STATUS "Threading: OpenMP ${OpenMP_VERSION} found")
else()
    set(NOVA_HAS_OPENMP 0)
    message(STATUS "Threading: OpenMP NOT found")
endif()
