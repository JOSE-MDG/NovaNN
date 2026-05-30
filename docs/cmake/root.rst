===================
Root CMakeLists.txt
===================

:file:  ``CMakeLists.txt`` (root)
:version: 5.0.0
:requires: CMake 3.27+, C23, C++23
:output: ``libnova.so``

Overview
--------

Top-level build configuration. Sets project metadata, language standards,
compile flags, backend selection options, and builds the ``nova`` shared
library.

Responsibilities
^^^^^^^^^^^^^^^^

1. Declares ``option(USE_CUDA ON)`` and ``option(USE_HIP ON)`` for GPU
   backend selection.
2. Includes the runtime-detection orchestrator (``NovaNNRuntime``).
3. Generates ``config.h`` from ``cmake/config.h.in``.
4. Defines common compile flags (``COMPILE_FLAGS``, ``DEBUG_FLAGS``,
   ``RELEASE_FLAGS``).
5. Builds ``libnova.so`` by merging two object libraries and
   whole-archiving the Rust memory crate.
6. Applies CPU / CUDA / HIP target configuration functions.
7. Sets preprocessor definitions ``NOVA_COMPILE_CUDA`` and
   ``NOVA_COMPILE_HIP`` based on the ``USE_*`` options.

Targets
-------

``nova`` — Shared Library
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Type:** SHARED
- **Output:** ``libnova.so``
- **Version:** 5.0.0 (SOVERSION 5)

Composition:

======================  ================================
Component               Source
======================  ================================
``ncore_core_obj``      C sources (object library)
``ncore_autograd_obj``  C++ autograd (object library)
``ncore_memory``        Rust memory crate (whole-archive)
======================  ================================

Linked libraries:

- ``${CMAKE_DL_LIBS}`` (``dl`` on Linux)
- ``m`` (math library)

Back-end configuration applied:

- ``novaNN_configure_cpu_target(nova)``
- ``novaNN_configure_cuda_target(nova)``
- ``novaNN_configure_hip_target(nova)``

Compile definitions:

- ``NOVA_COMPILE_CUDA=$<BOOL:${USE_CUDA}>``
- ``NOVA_COMPILE_HIP=$<BOOL:${USE_HIP}>``

Include directories:

- ``ncore/include`` (public)
- ``${CMAKE_BINARY_DIR}`` (private — for generated ``config.h``)

Configuration
-------------

Language Standards
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Language
     - Standard
     - Required
   * - C
     - C23
     - Yes
   * - C++
     - C++23
     - Yes

Backend Options
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``USE_CUDA``
     - ``ON``
     - Enable NVIDIA CUDA support
   * - ``USE_HIP``
     - ``ON``
     - Enable AMD HIP/ROCm support

When a backend option is ``OFF``, its corresponding
``novaNN_configure_*_target`` function becomes a no-op and the runtime
module reports "Disabled by user". CPU is always compiled as the mandatory
baseline.

Compile Flags
^^^^^^^^^^^^^

**Common flags** (all builds)::

   -Wall -Wextra -Wpedantic -Wshadow -Wcast-align -Wconversion
   -Wfloat-equal -Wformat=2 -Wimplicit-fallthrough -Wnull-dereference
   -Wpointer-arith -Wsign-conversion -Wundef -Wuninitialized -Wunused
   -Wno-missing-field-initializers -Wno-unused-parameter

**Debug flags**::

   -g -fno-omit-frame-pointer

**Release flags**::

   -O3 -march=native -ffast-math

**SIMD flags**: appended dynamically by ``DetectSIMD`` (see :doc:`DetectSIMD`).

Whole-Archive Linking
^^^^^^^^^^^^^^^^^^^^^

The Rust static library is linked with
``$<LINK_LIBRARY:WHOLE_ARCHIVE,ncore_memory>`` to guarantee that all FFI
symbols are visible to the Rust ``build.rs`` regardless of linker
garbage-collection.

Source
------

.. literalinclude:: ../../CMakeLists.txt
   :language: cmake
   :linenos:
