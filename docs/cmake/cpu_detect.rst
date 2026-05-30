==============================
CPU Instruction-Set Detectors
==============================

Individual modules that probe the compiler / CPU for specific SIMD
instruction-set extensions. Each uses the ``check_simd`` macro (defined in
:ref:`CheckInstructionSupport`) to compile and run a test snippet.

.. contents::
   :local:
   :depth: 1

--------

.. _CheckInstructionSupport:

CheckInstructionSupport
-----------------------

:file: ``cmake/utils/CheckInstructionSupport.cmake``

Provides the ``check_simd`` macro used by all CPU detector modules.

.. cmake:command:: check_simd

   Checks if the compiler supports a SIMD flag by attempting to compile
   and execute a test program.

   :param VAR:     Variable name to store result (1 = supported, 0 = not)
   :param FLAG:    Compiler flag to test (e.g., ``-mavx2``)
   :param SNIPPET: C++ code snippet to compile and run (must include ``main()``)

   Behaviour:

   1. Saves ``CMAKE_REQUIRED_FLAGS``.
   2. Appends ``FLAG``.
   3. Runs ``check_cxx_source_runs(SNIPPET VAR)``.
   4. Restores ``CMAKE_REQUIRED_FLAGS``.
   5. If ``VAR`` is true, appends ``FLAG`` to ``SIMD_FLAGS``.

   .. note:: ``check_simd`` is a **macro** (runs in caller scope). The
             temporary ``_saved_flags`` variable leaks into the caller —
             harmless (unlikely name collision).

   Example usage::

      check_simd(HAS_AVX2 "-mavx2" "
          #include <immintrin.h>
          int main() { __m256i a = _mm256_set1_epi32(1);
                       (void)_mm256_add_epi32(a, a); return 0; }
      ")

Source
~~~~~~

.. literalinclude:: ../../cmake/utils/CheckInstructionSupport.cmake
   :language: cmake
   :linenos:

--------

DetectSSE
---------

:file: ``cmake/detect/cpu/DetectSSE.cmake``

Probes SSE4.2 support via ``_mm_crc32_u32`` intrinsic.

Variables set:

- ``HAS_SSE4_2`` — 1 if SSE4.2 is supported

Source
~~~~~~

.. literalinclude:: ../../cmake/detect/cpu/DetectSSE.cmake
   :language: cmake
   :linenos:

--------

DetectAVX
---------

:file: ``cmake/detect/cpu/DetectAVX.cmake``

Probes AVX base, F16C, and FMA3. F16C and FMA3 are only tested if AVX
succeeds.

Variables set:

- ``HAS_AVX`` — 1 if AVX is supported
- ``HAS_F16C`` — 1 if F16C is supported (0 if no AVX)
- ``HAS_FMA3`` — 1 if FMA3 is supported (0 if no AVX)

Source
~~~~~~

.. literalinclude:: ../../cmake/detect/cpu/DetectAVX.cmake
   :language: cmake
   :linenos:

--------

DetectAVX2
----------

:file: ``cmake/detect/cpu/DetectAVX2.cmake``

Probes AVX2 base and AVX2-VNNI. VNNI is only tested if AVX2 succeeds;
on success ``HAS_AVX2_INT8`` is also set to 1.

Variables set:

- ``HAS_AVX2`` — 1 if AVX2 is supported
- ``HAS_AVX2_VNNI`` — 1 if AVX2-VNNI is supported
- ``HAS_AVX2_INT8`` — same as ``HAS_AVX2_VNNI`` (alias)

Source
~~~~~~

.. literalinclude:: ../../cmake/detect/cpu/DetectAVX2.cmake
   :language: cmake
   :linenos:

--------

DetectAVX512
------------

:file: ``cmake/detect/cpu/DetectAVX512.cmake``

Probes AVX-512 Foundation first, then six sub-extensions in dependency
order: BW, DQ, VL, VNNI, FP16, BF16. All sub-extensions are set to 0 if
AVX-512F is unsupported.

Variables set:

- ``HAS_AVX512F``
- ``HAS_AVX512_BW``
- ``HAS_AVX512_DQ``
- ``HAS_AVX512_VL``
- ``HAS_AVX512_VNNI``
- ``HAS_AVX512_FP16``
- ``HAS_AVX512_BF16``

Source
~~~~~~

.. literalinclude:: ../../cmake/detect/cpu/DetectAVX512.cmake
   :language: cmake
   :linenos:

--------

DetectAMX
---------

:file: ``cmake/detect/cpu/DetectAMX.cmake``

Probes AMX-Tile first, then AMX-FP16, AMX-BF16, and AMX-INT8. All
sub-extensions require AMX-Tile.

Variables set:

- ``HAS_AMX_TILE`` — 1 if AMX-Tile is supported
- ``HAS_AMX`` — same as ``HAS_AMX_TILE`` (aggregate flag)
- ``HAS_AMX_FP16``
- ``HAS_AMX_BF16``
- ``HAS_AMX_INT8``

Source
~~~~~~

.. literalinclude:: ../../cmake/detect/cpu/DetectAMX.cmake
   :language: cmake
   :linenos:
