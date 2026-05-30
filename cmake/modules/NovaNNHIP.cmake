#[[
    @file NovaNNHIP.cmake
    @brief HIP / ROCm back-end detection and target configuration.

    Detection:
      - Appends HIP_ROOT_DIR or ROCM_PATH (if set) to CMAKE_PREFIX_PATH.
      - Searches for HIP via find_package (QUIET CONFIG mode).
      - Rejects ROCm versions older than 6.2 (FATAL_ERROR).
      - On success: sets NOVA_HAS_HIP=1 and registers supported gfx targets.
      - Defines rejected gfx prefixes (Polaris, Vega, GCN, RDNA1) for
        validation in the configure function.

    @note The target configuration function is defined before backend
          detection so downstream CMakeLists can call it even when ROCm is
          unavailable; in that case it is a no-op.

    Target configuration (public function):

        novaNN_configure_hip_target(<target>)

      - No-op if NOVA_HAS_HIP is 0.
      - Validates AMDGPU_TARGETS (if user-overridden) against rejected
        prefixes; falls back to NOVA_HIP_ARCHITECTURES if unset.
      - Defines NOVA_HAS_HIP=1 and NOVA_ROCM_MIN_GFX=1030 on the target.
      - Links hip::host.
      - Sets HIP_STANDARD=17 and AMDGPU_TARGETS target property.

    Supported gfx list:
      - gfx908  (CDNA1)   — MI100
      - gfx90a  (CDNA2)   — MI200
      - gfx942  (CDNA3)   — MI300
      - gfx1030 (RDNA2)   — RX 6000 series
      - gfx1100 (RDNA3)   — RX 7000 series

    Rejected: gfx6xx (GCN1/2), gfx7xx (GCN3), gfx80x (Fiji/Polaris),
              gfx81x (Polaris), gfx900+ (Vega), gfx101x (RDNA1) —
              FATAL_ERROR at configure time if present in AMDGPU_TARGETS.

    Usage:
        include(modules/NovaNNHIP)            # detection
        novaNN_configure_hip_target(my_target) # per-target setup

    See also : NovaNNRuntime.cmake — orchestrator that includes this module
]]
function(novaNN_configure_hip_target TARGET)
    if(NOT NOVA_HAS_HIP)
        return()
    endif()
endfunction()

if(DEFINED HIP_ROOT_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${HIP_ROOT_DIR}")
endif()

if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
endif()

find_package(HIP QUIET CONFIG)

if(NOT HIP_FOUND)
    set(NOVA_HAS_HIP 0)
    message(STATUS "HIP: ROCm not found — HIP backend disabled")
    return()
endif()

set(NOVA_ROCM_MIN_VERSION "6.2")
if(HIP_VERSION VERSION_LESS NOVA_ROCM_MIN_VERSION)
    message(FATAL_ERROR
        "ROCm ${HIP_VERSION} is too old. "
        "NovaNN requires ROCm ${NOVA_ROCM_MIN_VERSION}+. "
        "Upgrade your ROCm installation or disable HIP."
    )
endif()

set(NOVA_HAS_HIP 1)
message(STATUS "HIP: ROCm ${HIP_VERSION} — ${ROCM_PATH}")

# Supported gfx targets
set(NOVA_HIP_ARCHITECTURES
    gfx908 # CDNA1  — MI100
    gfx90a # CDNA2  — MI200
    gfx942 # CDNA3  — MI300
    gfx1030 # RDNA2  — RX 6000
    gfx1100 # RDNA3  — RX 7000
)

# Unsupported gfx prefixes — caught at configure time if user passes
# -DAMDGPU_TARGETS manually
set(_NOVA_HIP_REJECTED_PREFIXES
    gfx6 # GCN1/2 (Polaris-era legacy)
    gfx7 # GCN2
    gfx80 # Fiji / Polaris (GCN3/4)
    gfx81 # Polaris
    gfx900 # Vega10
    gfx902 # Vega
    gfx904 # Vega
    gfx906 # Vega20 / Radeon VII
    gfx1010 # RDNA1 — Navi10
    gfx1011 # RDNA1
    gfx1012 # RDNA1
)

#[[
    @brief Configure a target for HIP compilation.
    @param TARGET CMake target name.
    No-op if HIP is unavailable.
]]
function(novaNN_configure_hip_target TARGET)
    if(NOT NOVA_HAS_HIP)
        return()
    endif()

    # Validate user-overridden AMDGPU_TARGETS if set
    if(DEFINED AMDGPU_TARGETS)
        foreach(GFX IN LISTS AMDGPU_TARGETS)
            foreach(REJECTED IN LISTS _NOVA_HIP_REJECTED_PREFIXES)
                if(GFX MATCHES "^${REJECTED}")
                    message(FATAL_ERROR
                        "GPU target '${GFX}' is not supported by NovaNN "
                        "(Polaris/Vega/GCN/RDNA1 are unsupported). "
                        "Remove it from AMDGPU_TARGETS."
                    )
                endif()
            endforeach()
        endforeach()
    else()
        set(AMDGPU_TARGETS "${NOVA_HIP_ARCHITECTURES}")
    endif()

    target_compile_definitions(${TARGET} PRIVATE
        NOVA_HAS_HIP=1
        NOVA_ROCM_MIN_GFX=1030 # RDNA2 consumer minimum
    )

    target_link_libraries(${TARGET} PRIVATE hip::host)

    set_target_properties(${TARGET} PROPERTIES
        HIP_STANDARD 17
        AMDGPU_TARGETS "${AMDGPU_TARGETS}"
    )
endfunction()
