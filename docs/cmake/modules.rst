===============
Runtime Modules
===============

Configuration modules that detect capabilities and provide
target-configuration functions. All are included (directly or indirectly) by
``NovaNNRuntime.cmake``.

.. contents::
   :local:
   :depth: 1

--------

NovaNNRuntime
-------------

:file: ``cmake/modules/NovaNNRuntime.cmake``

Runtime-detection orchestrator — single entry point for all feature
detection. Include in root ``CMakeLists.txt`` as::

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

1. ``detect/simd/DetectSIMD.cmake`` — CPU SIMD detection
2. ``modules/NovaNNCPU.cmake`` — CPU target configuration (SIMD + threading)
3. ``modules/NovaNNCUDA.cmake`` — CUDA detection + target configuration
   (guarded by ``USE_CUDA``; stub function when ``OFF``)
4. ``modules/NovaNNHIP.cmake`` — HIP detection + target configuration
   (guarded by ``USE_HIP``; stub function when ``OFF``)

**Backend gating.** The root ``CMakeLists.txt`` declares
``option(USE_CUDA ON)`` and ``option(USE_HIP ON)``. When a backend option is
``OFF``, NovaNNRuntime sets the corresponding ``NOVA_HAS_*`` variable to
``0`` and provides an empty stub for its configuration function, making the
entire backend a no-op. CPU is always included unconditionally.

Source
~~~~~~

.. literalinclude:: ../../cmake/modules/NovaNNRuntime.cmake
   :language: cmake
   :linenos:

--------

NovaNNCPU
---------

:file: ``cmake/modules/NovaNNCPU.cmake``

Applies CPU back-end capabilities to a target.

.. cmake:command:: novaNN_configure_cpu_target

   :param TARGET: CMake target name

   Applies to the target:

   1. ``SIMD_FLAGS`` — compile options for all detected extensions
   2. ``Threads::Threads`` — if ``NOVA_HAS_PTHREADS``
   3. ``OpenMP::OpenMP_C`` / ``OpenMP::OpenMP_CXX`` — if ``NOVA_HAS_OPENMP``
   4. ``NOVA_OPENMP=1`` or ``NOVA_OPENMP=0`` compile definition

   Usage::

      novaNN_configure_cpu_target(my_target)

   .. note:: The OpenMP target selection inspects the target's source files
             to determine whether to link ``OpenMP::OpenMP_C``,
             ``OpenMP::OpenMP_CXX``, or both.

Source
~~~~~~

.. literalinclude:: ../../cmake/modules/NovaNNCPU.cmake
   :language: cmake
   :linenos:

--------

NovaNNCUDA
----------

:file: ``cmake/modules/NovaNNCUDA.cmake``

Detects CUDA toolkit (>=12.6) and provides the target-configuration function.

Detection (runs once, guarded by ``if(DEFINED NOVA_HAS_CUDA)``):

- ``find_package(CUDAToolkit QUIET)``
- Rejects toolkits < 12.6 with ``FATAL_ERROR``
- Enables the CUDA language, sets ``NOVA_HAS_CUDA=1``
- Registers supported SM list: 75, 80, 86, 89, 90, 100

.. cmake:command:: novaNN_configure_cuda_target

   :param TARGET: CMake target name
   :no-op if: ``NOVA_HAS_CUDA`` is 0

   Applies to the target:

   1. ``NOVA_CUDA=1`` and ``NOVA_CUDA_MIN_SM=75`` definitions
   2. Links ``CUDA::cudart`` and ``CUDA::cuda_driver``
   3. Enables separable compilation
   4. Configures ``CUDA_ARCHITECTURES``:

      - If the user set ``CMAKE_CUDA_ARCHITECTURES``, validates that every
        SM is >= 75 (``FATAL_ERROR`` if any is below).
      - Otherwise, sets ``CUDA_ARCHITECTURES`` to ``native`` — compiles
        only for the local host GPU. This reduces compilation time and
        memory overhead in local development.

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

--------

NovaNNHIP
---------

:file: ``cmake/modules/NovaNNHIP.cmake``

Detects HIP / ROCm (>=6.2) and provides the target-configuration function.

Detection (runs once, guarded by ``if(DEFINED NOVA_HAS_HIP)``):

- Appends ``HIP_ROOT_DIR`` or ``ROCM_PATH`` to ``CMAKE_PREFIX_PATH``
- ``find_package(HIP QUIET CONFIG)``
- Rejects ROCm < 6.2 with ``FATAL_ERROR``
- Sets ``NOVA_HAS_HIP=1``
- Registers supported gfx targets and rejected prefixes

.. cmake:command:: novaNN_configure_hip_target

   :param TARGET: CMake target name
   :no-op if: ``NOVA_HAS_HIP`` is 0

   Applies to the target:

   1. Validates ``AMDGPU_TARGETS`` against rejected prefixes (defaults to
      ``NOVA_HIP_ARCHITECTURES`` if unset)
   2. ``NOVA_HIP=1`` and ``NOVA_ROCM_MIN_GFX=1030`` definitions
   3. Links ``hip::host``
   4. Sets ``HIP_STANDARD=17`` and ``AMDGPU_TARGETS``

   Supported gfx list:

   - gfx908  (CDNA1) — MI100
   - gfx90a  (CDNA2) — MI200
   - gfx942  (CDNA3) — MI300
   - gfx1030 (RDNA2) — RX 6000
   - gfx1100 (RDNA3) — RX 7000

   Rejected: gfx6xx (GCN1/2), gfx7xx (GCN3), gfx80x/81x (Polaris),
   gfx900/902/904/906 (Vega), gfx101x (RDNA1)

.. note:: Unlike its CUDA and threading siblings, this module does **not**
          have a multiple-inclusion guard. Currently included once via
          ``NovaNNRuntime.cmake``, so this is safe but inconsistent.

Source
~~~~~~

.. literalinclude:: ../../cmake/modules/NovaNNHIP.cmake
   :language: cmake
   :linenos:
