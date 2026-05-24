CMake Build System
===================

.. toctree::
   :maxdepth: 1
   :titlesonly:

   root
   ncore
   rust
   DetectSIMD
   cpu_detect
   threading
   modules
   config_h

Overview
--------

NovaNN uses **CMake 3.27+** as its build system.  The project is organised into
several layers:

**Core library** (``libncore.so``)
   C sources compiled as ``ncore_core_obj`` + C++ autograd as
   ``ncore_autograd_obj``, merged into a single shared library and
   whole-archived with the Rust memory crate.

**Rust / C++ FFI bridge**
   The ``rustcsrc`` static library (C++ FFI + device back-ends) is built first,
   then Cargo produces ``libncore_memory.a`` which links it.  CMake and Cargo
   communicate through environment variables set before invoking ``cargo build``.

**Runtime detection modules**
   SIMD capabilities, threading back-ends (pthreads, OpenMP), and optional GPU
   back-ends (CUDA, HIP) are detected at configure time and exposed via cmake
   functions that can be applied to any target.

Build Requirements
------------------

- CMake 3.27+
- C23 compatible compiler (GCC 14+, Clang 18+)
- C++23 compatible compiler
- Rust toolchain (edition 2024)
- Cargo

Optional (GPU back-ends):
- CUDA Toolkit 12.3+
- ROCm 6.2+

See Also
--------

- :doc:`root`           — top-level CMakeLists.txt
- :doc:`ncore`          — core library object builds
- :doc:`rust`           — Rust / C++ FFI bridge pipeline
- :doc:`DetectSIMD`     — SIMD detection orchestrator
- :doc:`cpu_detect`     — individual CPU instruction-set detectors
- :doc:`threading`      — POSIX threads and OpenMP detection
- :doc:`modules`        — runtime / CPU / CUDA / HIP configuration functions
- :doc:`config_h`       — generated config.h template
