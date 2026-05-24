Root CMakeLists.txt
-------------------

:file: ``CMakeLists.txt`` (root)
:version: 5.0.0
:requires: CMake 3.27+, C23, C++23
:output: ``libncore.so``

Overview
~~~~~~~~

Top-level build configuration that sets project metadata, C/C++ language
standards, compile flags, and builds the ``ncore`` shared library.

The root CMakeLists.txt is responsible for:

#. Including the runtime-detection orchestrator (``NovaNNRuntime``).
#. Generating ``config.h`` from ``cmake/config.h.in``.
#. Defining common compile flags (``COMPILE_FLAGS``, ``DEBUG_FLAGS``,
   ``RELEASE_FLAGS``).
#. Building ``libncore.so`` by merging two object libraries and
   whole-archiving the Rust memory crate.
#. Applying GPU back-end configuration (CUDA / HIP) to the final target.

Targets
~~~~~~~

``ncore`` — Shared Library
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Type:** SHARED
- **Output:** ``libncore.so``
- **Version:** 5.0.0 (SOVERSION 5)

Composition:

======================  ===================================
Component               Source
======================  ===================================
``ncore_core_obj``      C sources (object library)
``ncore_autograd_obj``  C++ autograd (object library)
``ncore_memory``        Rust memory crate (whole-archive)
======================  ===================================

Linked libraries:

- ``${CMAKE_DL_LIBS}`` (``dl`` on Linux)
- ``m`` (math library)

Back-end configuration applied:

- ``novaNN_configure_cpu_target(ncore)``
- ``novaNN_configure_cuda_target(ncore)``
- ``novaNN_configure_hip_target(ncore)``

Include directories:

- ``ncore/include`` (public)
- ``${CMAKE_BINARY_DIR}`` (private — for generated ``config.h``)

Configuration
~~~~~~~~~~~~~

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
~~~~~~

.. literalinclude:: ../../CMakeLists.txt
   :language: cmake
   :linenos:
