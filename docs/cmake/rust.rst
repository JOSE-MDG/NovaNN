Rust / C++ FFI Bridge
---------------------

:files: ``ncore/rust/CMakeLists.txt``, ``ncore/rust/csrc/CMakeLists.txt``,
        ``ncore/rust/build.rs``
:requires: Cargo, Rust toolchain (edition 2024)
:output: ``libncore_memory.a`` (via ``librustcsrc.a``)

Overview
~~~~~~~~

The Rust memory crate (``ncore_memory``) is the lowest-level memory manager.
It calls into a C++ FFI static library (``rustcsrc``) for GPU VRAM operations.

The build pipeline is:

#. ``csrc/CMakeLists.txt`` compiles the C++ FFI and device back-ends into
   ``librustcsrc.a``.
#. The parent ``CMakeLists.txt`` invokes ``cargo build`` via a custom target,
   passing the location of ``librustcsrc.a`` in the environment.
#. Cargo compiles the Rust crate and statically links ``librustcsrc.a``,
   producing ``libncore_memory.a``.
#. An ``IMPORTED STATIC`` target wraps the ``.a`` and exposes its link
   dependencies (pthreads, dl, m) transitively.

Files
~~~~~

.. list-table::
   :header-rows: 1

   * - File
     - Role
   * - ``ncore/rust/CMakeLists.txt``
     - Orchestrates the Cargo invocation and defines the imported target.
   * - ``ncore/rust/csrc/CMakeLists.txt``
     - Builds ``librustcsrc.a`` from C++ FFI + CUDA/HIP back-ends.
   * - ``ncore/rust/build.rs``
     - Rust build script that links ``librustcsrc.a`` and ``libstdc++``.
   * - ``ncore/rust/Cargo.toml``
     - Crate manifest (name = ``ncore_memory``, type = ``staticlib``).

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Set by CMake before ``cargo build``:

.. list-table::
   :header-rows: 1

   * - Variable
     - Value
     - Description
   * - ``RUSTFLAGS``
     - ``-C debuginfo=2 -C opt-level=0`` (Debug)
       ``-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C panic=abort`` (Release)
     - Flags passed to ``rustc``
   * - ``RUSTCSRC_DIR``
     - ``$<TARGET_FILE_DIR:rustcsrc>``
     - Directory containing ``librustcsrc.a``
   * - ``RUSTCSRC_NAME``
     - ``rustcsrc``
     - Library name (without prefix / extension)

C++ FFI Sources (``csrc/CMakeLists.txt``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always compiled:

- ``ffi.cpp`` — extern "C" bridge: reserve / release / memcpy wrappers
- ``device/admin.cpp`` — back-end admin: ``get_device_backend()``,
  ``device_reserve()``, ``device_release()``, ``device_memcpy()``

Conditionally compiled (guarded by ``NOVA_HAS_CUDA`` / ``NOVA_HAS_HIP``):

- ``device/cuda/cuda_allocator.cpp``, ``cuda_io.cpp``
- ``device/hip/hip_allocator.cpp``, ``hip_io.cpp``

build.rs Triggers
~~~~~~~~~~~~~~~~~

Cargo watches all ``csrc/`` files for changes via ``cargo:rerun-if-changed``.
Any modification to a C++ source or header triggers a rebuild of
``libncore_memory.a``.

Dependency Chain
~~~~~~~~~~~~~~~~

::

   csrc/CMakeLists.txt         →  librustcsrc.a
            ↓ (DEPENDS)
   ncore_memory_cargo (custom) →  cargo build → libncore_memory.a
            ↓ (add_dependencies)
   ncore_memory (IMPORTED)     →  linked into libncore.so (WHOLE_ARCHIVE)

Source: ``ncore/rust/CMakeLists.txt``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../ncore/rust/CMakeLists.txt
   :language: cmake
   :linenos:

Source: ``ncore/rust/csrc/CMakeLists.txt``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../ncore/rust/csrc/CMakeLists.txt
   :language: cmake
   :linenos:

Source: ``ncore/rust/build.rs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../ncore/rust/build.rs
   :language: rust
   :linenos:
