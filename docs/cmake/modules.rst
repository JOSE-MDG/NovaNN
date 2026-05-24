Runtime Modules
===============

Configuration modules that detect capabilities and provide target-configuration
functions.  All are included (directly or indirectly) by ``NovaNNRuntime.cmake``.

.. contents::
   :local:
   :depth: 1

----------

NovaNNRuntime
-------------

:file: ``cmake/modules/NovaNNRuntime.cmake``

Runtime-detection orchestrator — single entry point for all feature detection.
Include in root ``CMakeLists.txt`` as::

   include(modules/NovaNNRuntime)

After inclusion, the following are globally available:

- ``SIMD_FLAGS`` — compiler flags for enabled SIMD extensions
- ``HAS_*`` — per-feature capability flags
- ``NOVA_HAS_PTHREADS``
- ``NOVA_HAS_OPENMP``
- ``NOVA_HAS_CUDA``
- ``NOVA_HAS_HIP``
- ``novaNN_configure_cpu_target()``
- ``novaNN_configure_cuda_target()``
- ``novaNN_configure_hip_target()``

Include order:

#. ``detect/simd/DetectSIMD.cmake``
#. ``modules/NovaNNCPU.cmake``
#. ``modules/NovaNNCUDA.cmake``
#. ``modules/NovaNNHIP.cmake``

Source
~~~~~~

.. literalinclude:: ../../cmake/modules/NovaNNRuntime.cmake
   :language: cmake
   :linenos:

----------

NovaNNCPU
---------

:file: ``cmake/modules/NovaNNCPU.cmake``

Applies CPU back-end capabilities to a target.

.. cmake:command:: novaNN_configure_cpu_target

   :param TARGET: CMake target name

   Applies to the target:

   #. ``SIMD_FLAGS`` — compile options for all detected extensions
   #. ``Threads::Threads`` — if ``NOVA_HAS_PTHREADS``
   #. ``OpenMP::OpenMP_C`` / ``OpenMP::OpenMP_CXX`` — if ``NOVA_HAS_OPENMP``
   #. ``NOVA_OPENMP=1`` or ``NOVA_OPENMP=0`` compile definition

   Usage::

      novaNN_configure_cpu_target(my_target)

Source
~~~~~~

.. literalinclude:: ../../cmake/modules/NovaNNCPU.cmake
   :language: cmake
   :linenos:

----------

NovaNNCUDA
----------

:file: ``cmake/modules/NovaNNCUDA.cmake``

Detects CUDA toolkit (≥12.3) and provides the target-configuration function.

Detection (guarded, runs once):
   - ``find_package(CUDAToolkit QUIET)``
   - Rejects toolkits < 12.3 with ``FATAL_ERROR``
   - Enables the CUDA language, sets ``NOVA_HAS_CUDA=1``
   - Registers supported SM list: 75, 80, 86, 89, 90, 100

.. cmake:command:: novaNN_configure_cuda_target

   :param TARGET: CMake target name
   :no-op if: ``NOVA_HAS_CUDA`` is 0

   Applies to the target:

   #. ``NOVA_CUDA=1`` and ``NOVA_CUDA_MIN_SM=75`` definitions
   #. Links ``CUDA::cudart`` and ``CUDA::cuda_driver``
   #. Enables separable compilation
   #. Sets ``CUDA_ARCHITECTURES``
   #. Validates ``CMAKE_CUDA_ARCHITECTURES`` (any SM < 75 → ``FATAL_ERROR``)

   Supported SM list:

   - SM_75  (Turing) — RTX 2000
   - SM_80  (Ampere) — A100
   - SM_86  (Ampere) — RTX 3000 (consumer)
   - SM_89  (Ada) — RTX 4000
   - SM_90  (Hopper) — H100
   - SM_100 (Blackwell) — RTX 5000 / B100

   Rejected: Pascal (SM_60/61), Volta (SM_70)

Source
~~~~~~

.. literalinclude:: ../../cmake/modules/NovaNNCUDA.cmake
   :language: cmake
   :linenos:

----------

NovaNNHIP
---------

:file: ``cmake/modules/NovaNNHIP.cmake``

Detects HIP / ROCm (≥6.2) and provides the target-configuration function.

Detection:
   - Appends ``HIP_ROOT_DIR`` or ``ROCM_PATH`` to ``CMAKE_PREFIX_PATH``
   - ``find_package(HIP QUIET CONFIG)``
   - Rejects ROCm < 6.2 with ``FATAL_ERROR``
   - Sets ``NOVA_HAS_HIP=1``
   - Registers supported gfx targets and rejected prefixes

.. cmake:command:: novaNN_configure_hip_target

   :param TARGET: CMake target name
   :no-op if: ``NOVA_HAS_HIP`` is 0

   Applies to the target:

   #. Validates ``AMDGPU_TARGETS`` against rejected prefixes
      (defaults to ``NOVA_HIP_ARCHITECTURES`` if unset)
   #. ``NOVA_HIP=1`` and ``NOVA_ROCM_MIN_GFX=1030`` definitions
   #. Links ``hip::host``
   #. Sets ``HIP_STANDARD=17`` and ``AMDGPU_TARGETS``

   Supported gfx list:

   - gfx908  (CDNA1) — MI100
   - gfx90a  (CDNA2) — MI200
   - gfx942  (CDNA3) — MI300
   - gfx1030 (RDNA2) — RX 6000
   - gfx1100 (RDNA3) — RX 7000

   Rejected: gfx6xx (GCN1/2), gfx7xx (GCN3), gfx80x/81x (Polaris),
             gfx900/902/904/906 (Vega), gfx101x (RDNA1)

.. note:: Unlike its CUDA and threading siblings, this module does NOT have
          a multiple-inclusion guard.  Currently included once via
          ``NovaNNRuntime.cmake``, so this is safe but inconsistent.

Source
~~~~~~

.. literalinclude:: ../../cmake/modules/NovaNNHIP.cmake
   :language: cmake
   :linenos:
