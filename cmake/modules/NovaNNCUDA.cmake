#[[
    @file NovaNNCUDA.cmake
    @brief CUDA back-end detection and target configuration.

    Detection (runs once, guarded by `if(DEFINED NOVA_HAS_CUDA)`):
      - Searches for CUDAToolkit via find_package (QUIET).
      - Rejects toolkits older than 12.3 (FATAL_ERROR).
      - On success: enables the CUDA language, sets NOVA_HAS_CUDA=1, and
        registers supported SM architectures (75, 80, 86, 89, 90, 100).

    Target configuration (public function):

        novaNN_configure_cuda_target(<target>)

      - No-op if NOVA_HAS_CUDA is 0.
      - Defines NOVA_CUDA=1 and NOVA_CUDA_MIN_SM=75 on the target.
      - Links CUDA::cudart and CUDA::cuda_driver.
      - Enables separable compilation and sets CUDA_ARCHITECTURES.
      - Validates CMAKE_CUDA_ARCHITECTURES — any SM < 75 triggers FATAL_ERROR.

    Supported SM list:
      - SM_75  (Turing)     — RTX 2000 series
      - SM_80  (Ampere)     — A100
      - SM_86  (Ampere)     — RTX 3000 series (consumer)
      - SM_89  (Ada)        — RTX 4000 series
      - SM_90  (Hopper)     — H100
      - SM_100 (Blackwell)  — RTX 5000 / B100

    Rejected: Pascal (SM_60/61), Volta (SM_70) — FATAL_ERROR at configure time.

    Usage:
        include(modules/NovaNNCUDA)            # detection
        novaNN_configure_cuda_target(my_target) # per-target setup

    See also : NovaNNRuntime.cmake — orchestrator that includes this module
]]
function(novaNN_configure_cuda_target TARGET)
    if(NOT NOVA_HAS_CUDA)
        return()
    endif()
endfunction()

if(DEFINED NOVA_HAS_CUDA)
    return()
endif()

find_package(CUDAToolkit QUIET)

if(NOT CUDAToolkit_FOUND)
    set(NOVA_HAS_CUDA 0)
    message(STATUS "CUDA: toolkit not found — CUDA backend disabled")
    return()
endif()

# Version guard
set(NOVA_CUDA_MIN_VERSION "12.3")
if(CUDAToolkit_VERSION VERSION_LESS NOVA_CUDA_MIN_VERSION)
    message(FATAL_ERROR
        "CUDA ${CUDAToolkit_VERSION} is too old. "
        "NovaNN requires CUDA ${NOVA_CUDA_MIN_VERSION}+. "
        "Upgrade your CUDA toolkit or disable CUDA."
    )
endif()

enable_language(CUDA)
set(NOVA_HAS_CUDA 1)
message(STATUS "CUDA: ${CUDAToolkit_VERSION} — ${CUDAToolkit_LIBRARY_DIR}")

# Supported SM list (Pascal/Volta explicitly excluded)
set(NOVA_CUDA_ARCHITECTURES
    75 # Turing   — RTX 2000
    80 # Ampere   — A100, RTX 3000 (base)
    86 # Ampere   — RTX 3000 (consumer)
    89 # Ada      — RTX 4000
    90 # Hopper   — H100
    100 # Blackwell — RTX 5000 / B100
)

#[[
    @brief Configure a target for CUDA compilation.
    @param TARGET CMake target name.
    No-op if CUDA is unavailable.
]]
function(novaNN_configure_cuda_target TARGET)
    if(NOT NOVA_HAS_CUDA)
        return()
    endif()

    target_compile_definitions(${TARGET} PRIVATE
        NOVA_CUDA=1
        NOVA_CUDA_MIN_SM=75 # Turing minimum
    )

    target_link_libraries(${TARGET} PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
    )

    set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${NOVA_CUDA_ARCHITECTURES}"
    )

    # Reject builds targeting unsupported legacy SMs if someone
    # passes -DCMAKE_CUDA_ARCHITECTURES manually
    foreach(SM IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(SM LESS 75)
            message(FATAL_ERROR
                "CUDA SM ${SM} is not supported by NovaNN. "
                "Minimum is SM 75 (Turing). "
                "Remove SM ${SM} from CMAKE_CUDA_ARCHITECTURES."
            )
        endif()
    endforeach()
endfunction()
