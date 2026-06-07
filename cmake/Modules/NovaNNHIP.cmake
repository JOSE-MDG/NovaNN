#[=======================================================================[.rst:
.. module:: NovaNNHIP
   :synopsis: AMD HIP/ROCm backend detection and target configuration.

Provides the ``nova_configure_hip_target()`` function and handles
ROCm/HIP detection, version validation, and architecture
configuration.

This module is included by ``NovaNNRuntime.cmake`` only when
``USE_HIP`` is ``ON``.

**Detection logic:**

1. Searches for ROCm via ``$ENV{ROCM_PATH}`` or ``HIP_ROOT_DIR``.
2. Finds the HIP package via ``find_package(hip QUIET CONFIG)``.
3. Enforces a minimum ROCm version of **6.2**.
4. Enables the HIP language.

.. function:: nova_configure_hip_target(TARGET)

   Configure a target for the HIP backend.

   Applies the following to ``TARGET``:

   - Defines ``NOVA_HAS_HIP=1``.
   - Links ``hip::host``.
   - Sets HIP standard to C++23.
   - Validates architectures against a rejected-prefix list.

   :param TARGET: The target to configure (must already exist).
   :type TARGET:  ``target name``

   .. note::

      If ``NOVA_HAS_HIP`` is ``0`` (HIP not found or disabled), this
      function is a no-op.

   .. code-block:: cmake

      nova_configure_hip_target(mylib)

**Supported architectures:**

==========  =============================================
GFX ID      Architecture family
==========  =============================================
gfx908      CDNA2 (MI200 series)
gfx90a      CDNA3 (MI300 series)
gfx942      CDNA4 (MI400 series)
gfx1030     RDNA2 (RX 6000 series)
gfx1100     RDNA3 (RX 7000 series)
gfx1200     rdna4 (RX 9000 series)
==========  =============================================

.. warning::

   Legacy architectures (Polaris, Vega, RDNA1 — gfx6xx through
   gfx101x) are rejected with a ``FATAL_ERROR``.
#]=======================================================================]

if(WIN32)
    set(NOVA_HAS_HIP 0 CACHE INTERNAL "HIP backend availability")
    message(STATUS "HIP: not supported on Windows — HIP backend disabled")
    function(nova_configure_hip_target TARGET)
    endfunction()
    return()
endif()

function(nova_configure_hip_target TARGET)
    if(NOT NOVA_HAS_HIP)
        return()
    endif()

    if(DEFINED CMAKE_HIP_ARCHITECTURES)
        foreach(GFX IN LISTS CMAKE_HIP_ARCHITECTURES)
            foreach(REJECTED IN LISTS _NOVA_HIP_REJECTED_PREFIXES)
                if(GFX MATCHES "^${REJECTED}")
                    message(FATAL_ERROR
                        "GPU target '${GFX}' is not supported by NovaNN. "
                        "Legacy architectures (Polaris/Vega/RDNA1) are unsupported."
                    )
                endif()
            endforeach()
        endforeach()
        set_target_properties(${TARGET} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    else()
        set_target_properties(${TARGET} PROPERTIES HIP_ARCHITECTURES "${NOVA_SUPPORTED_HIP_ARCHS}")
    endif()

    target_compile_definitions(${TARGET} PRIVATE
        NOVA_HAS_HIP=1
    )

    target_link_libraries(${TARGET} PRIVATE hip::host)

    set_target_properties(${TARGET} PROPERTIES
        HIP_STANDARD 23
    )
endfunction()

if(NOVA_HAS_HIP)
    return()
endif()

if(DEFINED ENV{ROCM_PATH})
    set(NOVA_ROCM_PATH "$ENV{ROCM_PATH}")
    list(APPEND CMAKE_PREFIX_PATH "${NOVA_ROCM_PATH}")
elseif(DEFINED HIP_ROOT_DIR)
    set(NOVA_ROCM_PATH "${HIP_ROOT_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${NOVA_ROCM_PATH}")
endif()

find_package(hip QUIET CONFIG)

if(NOT hip_FOUND)
    set(NOVA_HAS_HIP 0 CACHE INTERNAL "HIP backend availability")
    message(STATUS "HIP: ROCm/HIP not found — HIP backend disabled")
    return()
endif()

set(NOVA_ROCM_MIN_VERSION "6.2")
if(hip_VERSION VERSION_LESS NOVA_ROCM_MIN_VERSION)
    message(FATAL_ERROR
        "ROCm ${hip_VERSION} is too old. "
        "NovaNN requires ROCm ${NOVA_ROCM_MIN_VERSION}+. "
        "Upgrade your ROCm installation or disable HIP."
    )
endif()

enable_language(HIP)

set(NOVA_HAS_HIP 1 CACHE INTERNAL "HIP backend availability")
message(STATUS "HIP: ROCm ${hip_VERSION} — ${NOVA_ROCM_PATH}")
set(NOVA_SUPPORTED_HIP_ARCHS gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1200)

set(_NOVA_HIP_REJECTED_PREFIXES
    gfx6 gfx7 gfx80 gfx81 gfx900 gfx902 gfx904 gfx906 gfx1010 gfx1011 gfx1012
)
