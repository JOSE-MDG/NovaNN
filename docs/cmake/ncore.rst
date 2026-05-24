ncore CMakeLists.txt
--------------------

:file: ``ncore/CMakeLists.txt``
:requires: Parent defines ``COMPILE_FLAGS``, ``SIMD_FLAGS``, ``DEBUG_FLAGS``,
           ``RELEASE_FLAGS``
:output: ``ncore_core_obj``, ``ncore_autograd_obj``

Overview
~~~~~~~~

Builds two object libraries consumed by the root CMakeLists.txt:

- ``ncore_core_obj`` — all C sources
- ``ncore_autograd_obj`` — all C++ autograd sources

Both targets inherit compile flags from the parent scope and have
``novaNN_configure_cpu_target()`` applied for SIMD / pthreads / OpenMP.

Targets
~~~~~~~

``ncore_core_obj`` — C Object Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Type:** OBJECT
- **Sources:** Discovered via ``GLOB_RECURSE`` from these patterns:

  * ``ncore/src/core/tables/*.c`` — ``dtype_tables.c``, ``cast_tables.c``,
    ``cast_dispatch_tables.c``
  * ``ncore/src/core/detect/*.c`` — ``cuda_device.c``, ``hip_device.c``
  * ``ncore/src/repr/*.c`` — all ``.c`` files in ``repr/`` subdirectories
    (formatting, layout, traversal, etc.)
  * ``ncore/src/core/*.c`` — ``alloc.c``, ``device.c``, ``tensor.c``,
    ``copy.c``, ``main.c``, ``dtype.c``, ``simd.c``

- **Includes:** ``ncore/include`` (public), ``ncore/src`` (private),
  ``${CMAKE_BINARY_DIR}``
- **Back-end:** ``novaNN_configure_cpu_target(ncore_core_obj)``

``ncore_autograd_obj`` — C++ Object Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Type:** OBJECT
- **Sources:** ``ncore/src/autograd/*.cpp`` (``tensor.cpp``, ``main.cpp``,
  ``node.cpp``, ``engine.cpp``)
- **Includes:** same as ``ncore_core_obj``
- **Back-end:** ``novaNN_configure_cpu_target(ncore_autograd_obj)``

Note on GLOB_RECURSE
^^^^^^^^^^^^^^^^^^^^

With ``file(GLOB_RECURSE ...)``, the ``*`` wildcard in patterns like
``ncore/src/repr/*.c`` matches path separators, so it finds ``.c`` files
in all subdirectories under ``repr/`` automatically.

Source
~~~~~~

.. literalinclude:: ../../ncore/CMakeLists.txt
   :language: cmake
   :linenos:
