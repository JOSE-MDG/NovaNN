#[=======================================================================[.rst:
.. module:: rust
   :synopsis: Rust memory subsystem build configuration.

Builds the ``ncore_memory`` Rust crate as a static library using Cargo
and makes it available as an imported CMake target.

**Targets:**

- ``memory_cargo`` — Custom target that runs ``cargo build`` for the
  ``ncore_memory`` crate.
- ``ncore_memory`` — IMPORTED STATIC library pointing to the Cargo
  output (``libncore_memory.a``).

**Build modes:**

- **Debug:** ``-C debuginfo=2 -C opt-level=0``
- **Release:** ``-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C panic=abort``

**Output:**

- **Linux:** ``libncore_memory.a``
- **Windows (MinGW/MSYS2):** ``libncore_memory.a``

**Environment variables forwarded to Cargo:**

- ``RUSTCSRC_DIR``  — Path to the compiled C++ FFI bridge (``rustcsrc``).
- ``RUSTCSRC_NAME`` — Name of the static library (``rustcsrc``).
- ``NOVA_HAS_CUDA`` — CUDA backend availability (``0`` or ``1``).
- ``NOVA_HAS_HIP``  — HIP backend availability (``0`` or ``1``).
- ``CARGO_TARGET_DIR`` — Cargo output directory.

**Dependencies:**

- ``rustcsrc`` — C++ FFI bridge from ``csrc/CMakeLists.txt``.
- ``Threads::Threads``, ``dl``, ``m`` — System libraries (linked as
  INTERFACE dependencies of ``ncore_memory``).

**Subdirectories:**

- ``csrc/`` — Builds the ``rustcsrc`` static library.
#]=======================================================================]

add_subdirectory(csrc)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RUSTFLAGS "-C debuginfo=2 -C opt-level=0")
    set(CARGO_PROFILE "")
    set(RUST_BUILD_SUBDIR "debug")
else()
    set(RUSTFLAGS "-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C panic=abort")
    set(CARGO_PROFILE "--release")
    set(RUST_BUILD_SUBDIR "release")
endif()

set(CARGO_TARGET_DIR "${CMAKE_BINARY_DIR}/cargo")
set(RUST_LIB "${CARGO_TARGET_DIR}/${RUST_BUILD_SUBDIR}/libncore_memory.a")

add_custom_target(memory_cargo
    COMMAND ${CMAKE_COMMAND} -E env
    "RUSTFLAGS=${RUSTFLAGS}"
    "RUSTCSRC_DIR=$<TARGET_FILE_DIR:rustcsrc>"
    "RUSTCSRC_NAME=rustcsrc"
    "NOVA_HAS_CUDA=${NOVA_HAS_CUDA}"
    "NOVA_HAS_HIP=${NOVA_HAS_HIP}"
    "CARGO_TARGET_DIR=${CARGO_TARGET_DIR}"
    cargo build ${CARGO_PROFILE}
    --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
    BYPRODUCTS ${RUST_LIB}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building Rust crate ncore_memory"
    DEPENDS rustcsrc
)

add_library(ncore_memory STATIC IMPORTED GLOBAL)
set_target_properties(ncore_memory PROPERTIES
    IMPORTED_LOCATION ${RUST_LIB}
)
add_dependencies(ncore_memory memory_cargo)

target_link_libraries(ncore_memory INTERFACE
    rustcsrc
    Threads::Threads
    ${CMAKE_DL_LIBS}
    m
)
