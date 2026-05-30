===========================
SIMD Detection — DetectSIMD
===========================

:file:  ``cmake/detect/simd/DetectSIMD.cmake``
:requires: CMake 3.15+, ``CheckCXXSourceRuns``
:output: ``SIMD_FLAGS``, all ``HAS_*`` capability variables

Overview
--------

Orchestrates SIMD instruction-set detection. Initialises ``SIMD_FLAGS``,
includes the shared :doc:`cpu_detect` utility, then runs every CPU-family
detector in dependency order:

1. ``DetectSSE.cmake`` — SSE4.2 (CRC32)
2. ``DetectAVX.cmake`` — AVX, F16C, FMA3
3. ``DetectAVX2.cmake`` — AVX2, AVX2-VNNI, AVX2-INT8
4. ``DetectAVX512.cmake`` — AVX-512F, BW, DQ, VL, VNNI, FP16, BF16
5. ``DetectAMX.cmake`` — AMX-Tile, AMX-FP16, AMX-BF16, AMX-INT8

Also sets the aggregate variable ``HAS_VNNI`` (true if either
``HAS_AVX2_VNNI`` or ``HAS_AVX512_VNNI`` is true).

Output Variables
----------------

.. cmake:variable:: SIMD_FLAGS

   Deduplicated list of ``-m`` flags for all detected SIMD extensions.
   Consumed by ``novaNN_configure_cpu_target()`` and by direct
   ``target_compile_options()`` calls.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Variable
     - Feature
     - Description
   * - ``HAS_SSE4_2``
     - SSE4.2
     - CRC32 instructions
   * - ``HAS_AVX``
     - AVX
     - 256-bit vector registers
   * - ``HAS_F16C``
     - F16C
     - FP16 conversion (requires AVX)
   * - ``HAS_FMA3``
     - FMA3
     - Fused multiply-add (requires AVX)
   * - ``HAS_AVX2``
     - AVX2
     - 256-bit integer SIMD
   * - ``HAS_AVX2_VNNI``
     - AVX2-VNNI
     - Integer dot-product (requires AVX2)
   * - ``HAS_AVX2_INT8``
     - AVX2-INT8
     - Same as AVX2-VNNI (alias)
   * - ``HAS_AVX512F``
     - AVX-512F
     - 512-bit vector foundation
   * - ``HAS_AVX512_BW``
     - AVX-512BW
     - Byte/word operations
   * - ``HAS_AVX512_DQ``
     - AVX-512DQ
     - Double/quadword operations
   * - ``HAS_AVX512_VL``
     - AVX-512VL
     - Vector length extension
   * - ``HAS_AVX512_VNNI``
     - AVX-512VNNI
     - Vector neural network
   * - ``HAS_AVX512_FP16``
     - AVX-512FP16
     - Half-precision floating point
   * - ``HAS_AVX512_BF16``
     - AVX-512BF16
     - Brain floating point
   * - ``HAS_AMX_TILE``
     - AMX-Tile
     - Advanced matrix extensions base
   * - ``HAS_AMX``
     - AMX (aggregate)
     - Same as ``HAS_AMX_TILE``
   * - ``HAS_AMX_FP16``
     - AMX-FP16
     - Tile FP16 (requires AMX-Tile)
   * - ``HAS_AMX_BF16``
     - AMX-BF16
     - Tile BF16 (requires AMX-Tile)
   * - ``HAS_AMX_INT8``
     - AMX-INT8
     - Tile INT8 (requires AMX-Tile)
   * - ``HAS_VNNI``
     - VNNI (aggregate)
     - AVX2-VNNI or AVX512-VNNI

Source
------

.. literalinclude:: ../../cmake/detect/simd/DetectSIMD.cmake
   :language: cmake
   :linenos:
