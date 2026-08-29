#[=======================================================================[.rst:
NCore Tests
-----------

Top-level test orchestrator for the NCore module.  Each subdirectory
under this tree produces one or more GoogleTest executables that are
automatically registered with CTest when :variable:`BUILD_TESTING` is
``ON``.

Subdirectories
^^^^^^^^^^^^^^

- ``dtypeCasting/`` — dtype casting tests (oracle utilities, cast
  table counts, kernel dispatch).  Registered in W0 of the rewrite.
- ``dtypes/`` — reduced-precision dtype tests.
- ``core/`` — core runtime tests (status propagation, device
  detection, memory allocator, tensor copies/transfers).
#]=======================================================================]

if(BUILD_TESTING)
    # Sanitizer runtime configuration for GPU interop (see suppr.txt at the
    # repository root).
    #
    # ASAN_OPTIONS: protect_shadow_gap=0 is required for CUDA device probes
    # inside an ASan process; without it cudaGetDeviceCount fails and every
    # device-memory test reports "device kind is not recognized".
    #
    # LSAN_OPTIONS: discards leaks internal to the NVIDIA driver and the HSA
    # runtime (third-party caches freed at process exit by their owners).
    set(NOVA_SANITIZER_ENV "")
    if(USE_ASAN)
        # protect_shadow_gap=0 is required for CUDA probes on Linux, but
        # crashes Windows ASan (LLVM 22 CHECK in sanitizer_common_libcdep.cpp).
        if(NOT WIN32)
            list(APPEND NOVA_SANITIZER_ENV "ASAN_OPTIONS=protect_shadow_gap=0")
            # LSAN suppressions (suppr.txt contains libcuda.so etc.).
            list(APPEND NOVA_SANITIZER_ENV "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/suppr.txt")
        endif()
    endif()

    add_subdirectory(core)
    add_subdirectory(dtypeCasting)
    add_subdirectory(dtypes)
endif()
