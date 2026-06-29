#[=======================================================================[.rst:
NovaNNHIP
---------

AMD HIP/ROCm backend detection and target configuration for the NovaNN
project.  Detects the ROCm installation, enables the HIP language, and
provides functions to configure HIP-specific target properties.

If ``NOVA_HAS_HIP`` is already defined the module returns immediately
(idempotent guard).  On Windows the module disables the HIP backend
and provides empty stub functions.

This module defines the following cache variables:

``NOVA_HAS_HIP``
  ``1`` when a supported ROCm installation is found, ``0`` otherwise.

``NOVA_SUPPORTED_HIP_ARCHS``
  Default list of supported HIP GPU architectures:
  gfx908 gfx90a gfx942 gfx950
  gfx1030
  gfx1100 gfx1101
  gfx1200 gfx1201.

This module defines the following functions:

.. command:: nova_configure_hip_target

  Configure a target with HIP compile definitions:

  .. code-block:: cmake

    nova_configure_hip_target(<target>)

.. command:: nova_configure_hip_runtime_target

  Configure a target that links against the HIP runtime:

  .. code-block:: cmake

    nova_configure_hip_runtime_target(<target>)

.. command:: nova_configure_hip_kernels_target

  Configure a target that compiles HIP kernel code:

  .. code-block:: cmake

    nova_configure_hip_kernels_target(<target> [EXTRA_LIBS <lib> ...])

#]=======================================================================]

if(WIN32)
    set(NOVA_HAS_HIP 0 CACHE INTERNAL "HIP backend availability")
    message(STATUS "HIP: not supported on Windows — HIP backend disabled")

    function(nova_configure_hip_target TARGET)
    endfunction()

    function(nova_configure_hip_runtime_target TARGET)
    endfunction()

    function(nova_configure_hip_kernels_target TARGET)
    endfunction()
    return()
endif()

set(NOVA_SUPPORTED_HIP_ARCHS

    # CDNA1 (Arcturus)
    gfx908

    # CDNA2 (Aldebaran)
    gfx90a

    # CDNA3 (Aqua Vanjaram)
    gfx942

    # CDNA4 (CDNA Next)
    gfx950

    # RDNA2
    gfx1030

    # RDNA3
    gfx1100 gfx1101

    # RDNA4
    gfx1200 gfx1201
)

set(_NOVA_HIP_REJECTED_PREFIXES
    gfx6 gfx7 gfx80 gfx81 gfx900 gfx902 gfx904 gfx906 gfx101 gfx1011 gfx1012
)

#[=======================================================================[.rst:
.. command:: _nova_configure_hip_macros

  Internal helper that defines the ``NOVA_HAS_HIP`` preprocessor
  macro on the given target:

  .. code-block:: cmake

    _nova_configure_hip_macros(<target>)

#]=======================================================================]
function(_nova_configure_hip_macros TARGET)
    if(NOT NOVA_HAS_HIP)
        return()
    endif()

    target_compile_definitions(${TARGET} PRIVATE
        NOVA_HAS_HIP=1
    )
endfunction()

#[=======================================================================[.rst:
.. command:: nova_configure_hip_target

  Configure a target with HIP compile definitions:

  .. code-block:: cmake

    nova_configure_hip_target(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function adds the ``NOVA_HAS_HIP=1`` compile definition when
  the HIP backend is available.  It does not link against the HIP
  runtime; use ``nova_configure_hip_runtime_target`` for that.

#]=======================================================================]
function(nova_configure_hip_target TARGET)
    if(NOT NOVA_HAS_HIP)
        return()
    endif()

    _nova_configure_hip_macros(${TARGET})
endfunction()

#[=======================================================================[.rst:
.. command:: _nova_configure_hip_common

  Internal helper that applies common HIP target properties:

  .. code-block:: cmake

    _nova_configure_hip_common(<target>)

  Validates ``CMAKE_HIP_ARCHITECTURES`` against
  ``_NOVA_HIP_REJECTED_PREFIXES`` (legacy Polaris/Vega/RDNA1
  architectures).  When no user-specified architectures are provided
  the function defaults to ``NOVA_SUPPORTED_HIP_ARCHS``.  Sets
  ``HIP_STANDARD`` to 23 and defines ``NOVA_HAS_HIP=1``.

#]=======================================================================]
function(_nova_configure_hip_common TARGET)
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
    else()
        set_target_properties(${TARGET} PROPERTIES
            HIP_STANDARD 23
            HIP_ARCHITECTURES "${NOVA_SUPPORTED_HIP_ARCHS}"
        )
    endif()

    _nova_configure_hip_macros(${TARGET})

    set_target_properties(${TARGET} PROPERTIES
        HIP_STANDARD 23
    )
endfunction()

#[=======================================================================[.rst:
.. command:: nova_configure_hip_runtime_target

  Configure a target that links against the HIP runtime:

  .. code-block:: cmake

    nova_configure_hip_runtime_target(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function calls ``_nova_configure_hip_common`` and links the
  target against ``hip::host``.

#]=======================================================================]
function(nova_configure_hip_runtime_target TARGET)
    if(NOT NOVA_HAS_HIP)
        return()
    endif()

    _nova_configure_hip_common(${TARGET})

    target_link_libraries(${TARGET} PRIVATE
        hip::host
    )
endfunction()

#[=======================================================================[.rst:
.. command:: nova_configure_hip_kernels_target

  Configure a target that compiles HIP kernel code:

  .. code-block:: cmake

    nova_configure_hip_kernels_target(<target> [EXTRA_LIBS <lib> ...])

  The ``<target>`` argument specifies the CMake target to configure.

  Any additional arguments after ``<target>`` are forwarded as
  private link libraries to the target.

#]=======================================================================]
function(nova_configure_hip_kernels_target TARGET)
    if(NOT NOVA_HAS_HIP)
        return()
    endif()

    _nova_configure_hip_common(${TARGET})

    if(ARGN)
        target_link_libraries(${TARGET} PRIVATE ${ARGN})
    endif()
endfunction()

if(DEFINED ENV{ROCM_PATH})
    set(NOVA_ROCM_PATH "$ENV{ROCM_PATH}")
    list(APPEND CMAKE_PREFIX_PATH "${NOVA_ROCM_PATH}")
elseif(DEFINED HIP_ROOT_DIR)
    set(NOVA_ROCM_PATH "${HIP_ROOT_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${NOVA_ROCM_PATH}")
endif()

find_package(hip QUIET CONFIG)

if(NOVA_HAS_HIP)
    return()
endif()

if(NOT hip_FOUND)
    set(NOVA_HAS_HIP 0 CACHE INTERNAL "HIP backend availability")
    message(STATUS "HIP: ROCm/HIP not found — HIP backend disabled")
    return()
endif()

set(NOVA_ROCM_MIN_VERSION "7.0")

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
