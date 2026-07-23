#[=======================================================================[.rst:
NovaNNCUDA
----------

NVIDIA CUDA backend detection and target configuration for the NovaNN
project.  Detects the CUDA toolkit, enables the CUDA language, and
provides functions to configure CUDA-specific target properties.

If ``NOVA_HAS_CUDA`` is already defined the module returns immediately
(idempotent guard).

This module defines the following cache variables:

``NOVA_HAS_CUDA``
  ``1`` when a supported CUDA toolkit is found, ``0`` otherwise.

``NOVA_CUDA_ARCHITECTURES``
  Default list of CUDA SM architectures:
  75, 80, 86, 89, 90, 100, 103, 110, 120, 121.

This module defines the following functions:

.. command:: nova_configure_cuda_target

  Configure a target with CUDA compile definitions:

  .. code-block:: cmake

    nova_configure_cuda_target(<target>)

.. command:: nova_configure_cuda_runtime_target

  Configure a target that links against the CUDA runtime:

  .. code-block:: cmake

    nova_configure_cuda_runtime_target(<target>)


.. command:: nova_configure_cuda_includes_target

  Configure a target that include the CUDA headers:

  .. code-block:: cmake

    nova_configure_cuda_includes_target(<target>)

.. command:: nova_configure_cuda_kernels_target

  Configure a target that compiles CUDA kernel code:

  .. code-block:: cmake

    nova_configure_cuda_kernels_target(<target> [EXTRA_LIBS <lib> ...])

#]=======================================================================]

#[=======================================================================[.rst:
.. command:: _nova_configure_cuda_common

  Internal helper that applies common CUDA target properties:

  .. code-block:: cmake

    _nova_configure_cuda_common(<target>)

  Sets ``CUDA_STANDARD`` to 20, enables separable compilation for
  non-OBJECT libraries, validates ``CMAKE_CUDA_ARCHITECTURES``
  (minimum SM 75), and defines ``NOVA_HAS_CUDA=1``.

#]=======================================================================]
function(_nova_configure_cuda_common TARGET)
    if(NOT NOVA_HAS_CUDA)
        return()
    endif()

    _nova_configure_cuda_macros(${TARGET})

    set_target_properties(${TARGET} PROPERTIES
        CUDA_STANDARD 20
    )

    get_target_property(_target_type ${TARGET} TYPE)

    if(NOT _target_type STREQUAL "OBJECT_LIBRARY")
        set_target_properties(${TARGET} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()

    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        foreach(SM IN LISTS CMAKE_CUDA_ARCHITECTURES)
            if(SM MATCHES "^[0-9]+$")
                if(SM LESS 75)
                    message(FATAL_ERROR
                        "CUDA SM ${SM} is not supported by NovaNN. "
                        "Minimum is SM 75 (Turing). "
                        "Remove SM ${SM} from CMAKE_CUDA_ARCHITECTURES."
                    )
                endif()
            endif()
        endforeach()
    else()
        set_target_properties(${TARGET} PROPERTIES
            CUDA_ARCHITECTURES "${NOVA_CUDA_ARCHITECTURES}"
        )
    endif()
endfunction()

#[=======================================================================[.rst:
.. command:: _nova_configure_cuda_macros

  Internal helper that defines the ``NOVA_HAS_CUDA`` preprocessor
  macro on the given target:

  .. code-block:: cmake

    _nova_configure_cuda_macros(<target>)

#]=======================================================================]
function(_nova_configure_cuda_macros TARGET)
    if(NOT NOVA_HAS_CUDA)
        return()
    endif()

    target_compile_definitions(${TARGET} PRIVATE
        NOVA_HAS_CUDA=1
    )
endfunction()

#[=======================================================================[.rst:
.. command:: nova_configure_cuda_target

  Configure a target with CUDA compile definitions:

  .. code-block:: cmake

    nova_configure_cuda_target(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function adds the ``NOVA_HAS_CUDA=1`` compile definition when
  the CUDA backend is available.  It does not link against the CUDA
  runtime; use ``nova_configure_cuda_runtime_target`` for that.

#]=======================================================================]
function(nova_configure_cuda_target TARGET)
    if(NOT NOVA_HAS_CUDA)
        return()
    endif()

    _nova_configure_cuda_macros(${TARGET})
endfunction()

#[=======================================================================[.rst:
.. command:: nova_configure_cuda_runtime_target

  Configure a target that links against the CUDA runtime:

  .. code-block:: cmake

    nova_configure_cuda_runtime_target(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function calls ``_nova_configure_cuda_common`` and links the
  target against ``CUDA::cudart``.

#]=======================================================================]
function(nova_configure_cuda_runtime_target TARGET)
    if(NOT NOVA_HAS_CUDA)
        return()
    endif()

    _nova_configure_cuda_common(${TARGET})

    target_link_libraries(${TARGET} PRIVATE
        CUDA::cudart
    )
endfunction()

#[=======================================================================[.rst:
.. command:: nova_configure_cuda_includes_target

  Configure a target that include the CUDA headers:

  .. code-block:: cmake

    nova_configure_cuda_includes_target(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function calls ``_nova_configure_cuda_common`` and include
  the ``${CUDAToolkit_INCLUDE_DIRS}`` headers.

#]=======================================================================]
function(nova_configure_cuda_includes_target TARGET)
    if(NOT NOVA_HAS_CUDA)
        return()
    endif()

    _nova_configure_cuda_common(${TARGET})

    target_include_directories(${TARGET} INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
endfunction()

#[=======================================================================[.rst:
.. command:: nova_configure_cuda_kernels_target

  Configure a target that compiles CUDA kernel code:

  .. code-block:: cmake

    nova_configure_cuda_kernels_target(<target> [EXTRA_LIBS <lib> ...])

  The ``<target>`` argument specifies the CMake target to configure.

  Additional ``EXTRA_LIBS`` are parsed as a keyword argument and linked
  as private libraries.

#]=======================================================================]
function(nova_configure_cuda_kernels_target TARGET)
    if(NOT NOVA_HAS_CUDA)
        return()
    endif()

    cmake_parse_arguments(_CUDA_K "" "" "EXTRA_LIBS" ${ARGN})

    _nova_configure_cuda_common(${TARGET})

    if(_CUDA_K_EXTRA_LIBS)
        target_link_libraries(${TARGET} PRIVATE ${_CUDA_K_EXTRA_LIBS})
    endif()
endfunction()

find_package(CUDAToolkit QUIET)

if(NOVA_HAS_CUDA)
    return()
endif()

if(NOT CUDAToolkit_FOUND)
    set(NOVA_HAS_CUDA 0)
    message(STATUS "CUDA: toolkit not found — CUDA backend disabled")
    return()
endif()

set(NOVA_CUDA_MIN_VERSION "13.0")

if(CUDAToolkit_VERSION VERSION_LESS NOVA_CUDA_MIN_VERSION)
    message(FATAL_ERROR
        "CUDA ${CUDAToolkit_VERSION} is too old. "
        "NovaNN requires CUDA ${NOVA_CUDA_MIN_VERSION}+. "
        "Upgrade your CUDA toolkit or disable CUDA."
    )
endif()

set(NOVA_HAS_CUDA 1 CACHE INTERNAL "CUDA backend availability")
message(STATUS "CUDA: ${CUDAToolkit_VERSION} — ${CUDAToolkit_LIBRARY_DIR}")

set(NOVA_CUDA_ARCHITECTURES

    # Turing
    75

    # Ampere
    80
    86

    # Ada Lovelace
    89

    # Hopper
    90

    # Blackwell – Data Center (B100/B200: sm_100; B300/GB300 Ultra: sm_103)
    100
    103

    # Blackwell – Jetson Thor (renamed from sm_101 in CUDA 13.0+)
    110

    # Blackwell – Consumer/Workstation (RTX 50-series: sm_120; GB10/DGX Spark: sm_121)
    120
    121
)
