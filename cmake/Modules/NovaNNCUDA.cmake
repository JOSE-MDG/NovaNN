#[=======================================================================[.rst:
.. module:: NovaNNCUDA
   :synopsis: NVIDIA CUDA backend detection and target configuration.

Provides the ``nova_configure_cuda_target()`` function and handles
CUDA toolkit detection, version validation, and architecture
configuration.

This module is included by ``NovaNNRuntime.cmake`` only when
``USE_CUDA`` is ``ON``.

**Detection logic:**

1. Searches for the CUDA toolkit via ``find_package(CUDAToolkit)``.
2. Enforces a minimum CUDA toolkit version of **12.6**.
3. Enables the CUDA language and sets default architecture targets:
   SM 75 (Turing), 80, 86, 89, 90, 100 (Blackwell).

.. function:: nova_configure_cuda_target(TARGET)

   Configure a target for the CUDA backend.

   Applies the following to ``TARGET``:

   - Defines ``NOVA_HAS_CUDA=1``.
   - Links ``CUDA::cudart``.
   - Sets CUDA standard to C++23.
   - Enables CUDA separable compilation.
   - Validates and sets CUDA architectures (minimum SM 75).

   :param TARGET: The target to configure (must already exist).
   :type TARGET:  ``target name``

   .. note::

      If ``NOVA_HAS_CUDA`` is ``0`` (CUDA not found or disabled), this
      function is a no-op.

   .. code-block:: cmake

      nova_configure_cuda_target(mylib)

**Supported architectures:**

=========  ==========  =============================================
SM         Generation  Notable hardware
=========  ==========  =============================================
75         Turing      RTX 2000 series / T4
80         Ampere      A100, RTX 3000 (base)
86         Ampere      RTX 3000 (consumer)
89         Ada         RTX 4000 series
90         Hopper      H100
100        Blackwell   RTX 5000 / B100 / B200
=========  ==========  =============================================

.. warning::

   Architectures below SM 75 (Turing) are rejected with a
   ``FATAL_ERROR``.  Remove them from ``CMAKE_CUDA_ARCHITECTURES``.
#]=======================================================================]

function(nova_configure_cuda_target TARGET)
    if(NOT NOVA_HAS_CUDA)
        return()
    endif()

    target_compile_definitions(${TARGET} PRIVATE
        NOVA_HAS_CUDA=1
    )

    target_link_libraries(${TARGET} PRIVATE
        CUDA::cudart
    )

    set_target_properties(${TARGET} PROPERTIES
        HIP_STANDARD 23
        CUDA_SEPARABLE_COMPILATION ON
    )

    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        foreach(SM IN LISTS CMAKE_CUDA_ARCHITECTURES)
            if(SM LESS 75)
                message(FATAL_ERROR
                    "CUDA SM ${SM} is not supported by NovaNN. "
                    "Minimum is SM 75 (Turing). "
                    "Remove SM ${SM} from CMAKE_CUDA_ARCHITECTURES."
                )
            endif()
        endforeach()

        set_target_properties(${TARGET} PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )
    else()
        set_target_properties(${TARGET} PROPERTIES
            CUDA_ARCHITECTURES "${NOVA_CUDA_ARCHITECTURES}"
        )
    endif()
endfunction()

if(NOVA_HAS_CUDA)
    return()
endif()

find_package(CUDAToolkit QUIET)

if(NOT CUDAToolkit_FOUND)
    set(NOVA_HAS_CUDA 0)
    message(STATUS "CUDA: toolkit not found — CUDA backend disabled")
    return()
endif()

set(NOVA_CUDA_MIN_VERSION "12.6")
if(CUDAToolkit_VERSION VERSION_LESS NOVA_CUDA_MIN_VERSION)
    message(FATAL_ERROR
        "CUDA ${CUDAToolkit_VERSION} is too old. "
        "NovaNN requires CUDA ${NOVA_CUDA_MIN_VERSION}+. "
        "Upgrade your CUDA toolkit or disable CUDA."
    )
endif()

enable_language(CUDA)
set(NOVA_HAS_CUDA 1 CACHE INTERNAL "CUDA backend availability")
message(STATUS "CUDA: ${CUDAToolkit_VERSION} — ${CUDAToolkit_LIBRARY_DIR}")

set(NOVA_CUDA_ARCHITECTURES
    75 # Turing   — RTX 2000 / T4
    80 # Ampere   — A100, RTX 3000 (base)
    86 # Ampere   — RTX 3000 (consumer)
    89 # Ada      — RTX 4000
    90 # Hopper   — H100
    100 # Blackwell — RTX 5000 / B100 / B200
)
