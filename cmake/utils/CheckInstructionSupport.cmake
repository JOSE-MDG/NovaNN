#[[
    @file CheckInstructionSupport.cmake
    @brief Generic macro for testing compiler / CPU instruction-set support.

    Provides the check_simd macro used by all Detect*.cmake modules to probe
    whether the compiler can compile *and* execute a given SIMD snippet under
    a specific compiler flag.

    Macro: check_simd(VAR FLAG SNIPPET)

      @param VAR     — Variable name to store result (1 = supported, 0 = not)
      @param FLAG    — Compiler flag to test (e.g. -mavx2)
      @param SNIPPET — C++ source snippet to compile and run (must include main())

    Behaviour:
      1. Saves CMAKE_REQUIRED_FLAGS.
      2. Appends FLAG to CMAKE_REQUIRED_FLAGS.
      3. Runs check_cxx_source_runs(SNIPPET VAR).
      4. Restores CMAKE_REQUIRED_FLAGS.
      5. If VAR is true, appends FLAG to the global SIMD_FLAGS list.

    @note Because check_simd is a **macro** (not a function), it runs in the
          caller's scope.  The temporary _saved_flags variable leaks into the
          caller scope — this is harmless since the name is unlikely to collide.

    Requires : CMake 3.15+, CheckCXXSourceRuns
    See also : DetectSIMD.cmake — orchestrator that uses this macro
               NovaNNCPU.cmake  — consumer of the resulting SIMD_FLAGS
]]

include(CheckCXXSourceRuns)

#[[
    @brief Check if compiler supports a SIMD flag.

    Attempts to compile and run a snippet with the given flag. On success,
    appends the flag to SIMD_FLAGS and sets VAR to 1.

    @param VAR    Variable name to store result (1=supported, 0=not)
    @param FLAG   Compiler flag to test (e.g., -mavx2)
    @param SNIPPET C++ code snippet to compile and execute
]]
macro(check_simd VAR FLAG SNIPPET)
    set(_saved_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${FLAG}")
    check_cxx_source_runs("${SNIPPET}" ${VAR})
    set(CMAKE_REQUIRED_FLAGS "${_saved_flags}")
    if(${VAR})
        list(APPEND SIMD_FLAGS "${FLAG}")
    endif()
endmacro()
