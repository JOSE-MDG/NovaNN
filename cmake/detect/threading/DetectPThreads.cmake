#[[
    @file DetectPThreads.cmake
    @brief POSIX threads detection.

    Uses `find_package(Threads REQUIRED)` to detect pthreads support.
    On success (CMAKE_USE_PTHREADS_INIT is true):
      - Sets NOVA_HAS_PTHREADS = 1
    Otherwise:
      - Sets NOVA_HAS_PTHREADS = 0 (generic Threads fallback)

    Multiple-inclusion guard: returns early if NOVA_HAS_PTHREADS is already
    defined, so it is safe to include from multiple call sites.

    Variables set:
      - NOVA_HAS_PTHREADS  — 1 if pthreads is available, 0 otherwise

    Targets exposed:
      - Threads::Threads   — imported target (always available after find_package)

    Usage:
        include(detect/threading/DetectPThreads)

    See also : NovaNNCPU.cmake — consumes NOVA_HAS_PTHREADS for target config
]]
if(DEFINED NOVA_HAS_PTHREADS)
    return()
endif()

find_package(Threads REQUIRED)

if(CMAKE_USE_PTHREADS_INIT)
    set(NOVA_HAS_PTHREADS 1)
    message(STATUS "Threading: pthreads found")
else()
    set(NOVA_HAS_PTHREADS 0)
    message(STATUS "Threading: pthreads NOT found (using generic Threads)")
endif()
