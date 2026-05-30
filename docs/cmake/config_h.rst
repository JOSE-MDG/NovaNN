=================
config.h Template
=================

:file:  ``cmake/config.h.in``
:output: ``${CMAKE_BINARY_DIR}/config.h``

Overview
--------

CMake configuration template consumed by ``configure_file()``. Every
``@VAR@`` placeholder is substituted with the corresponding CMake variable
at configure time. The resulting ``config.h`` is included by C/C++ source
files (via private include path) to conditionally enable SIMD paths and
feature gates.

The template exposes the project version and all ``HAS_*`` capability flags
set by the :doc:`DetectSIMD` and :doc:`cpu_detect` modules.

Variables
---------

.. list-table::
   :header-rows: 1

   * - Macro
     - CMake Source Variable
     - Description
   * - ``VERSION_MAJOR``
     - ``PROJECT_VERSION_MAJOR``
     - Project major version
   * - ``VERSION_MINOR``
     - ``PROJECT_VERSION_MINOR``
     - Project minor version
   * - ``VERSION_PATCH``
     - ``PROJECT_VERSION_PATCH``
     - Project patch version
   * - ``HAS_SSE4_2``
     - ``HAS_SSE4_2``
     - SSE4.2 support
   * - ``HAS_AVX``
     - ``HAS_AVX``
     - AVX support
   * - ``HAS_AVX2``
     - ``HAS_AVX2``
     - AVX2 support
   * - ``HAS_AVX2_INT8``
     - ``HAS_AVX2_INT8``
     - AVX2-INT8 support
   * - ``HAS_AVX2_VNNI``
     - ``HAS_AVX2_VNNI``
     - AVX2-VNNI support
   * - ``HAS_F16C``
     - ``HAS_F16C``
     - F16C support
   * - ``HAS_VNNI``
     - ``HAS_VNNI``
     - Aggregate VNNI flag
   * - ``HAS_FMA3``
     - ``HAS_FMA3``
     - FMA3 support
   * - ``HAS_AVX512F`` through ``HAS_AVX512_BF16``
     - Corresponding ``HAS_AVX512_*``
     - AVX-512 sub-extensions
   * - ``HAS_AMX`` through ``HAS_AMX_INT8``
     - Corresponding ``HAS_AMX_*``
     - AMX extensions

Source
------

.. literalinclude:: ../../cmake/config.h.in
   :language: c
   :linenos:
