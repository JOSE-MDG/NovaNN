#[=======================================================================[.rst:
Rust Crate Build
----------------

Build the ``ncore_memory`` Rust crate via Cargo and expose it as an
imported static library.  The Cargo build is orchestrated through a
custom target that passes all required environment variables to
``cargo build``.

Variables defined:

``RUST_LIB``
  Path to the built static library (``libncore_memory.a``).

``CARGO_TARGET_DIR``
  Cargo output directory inside the CMake build tree.

Targets created:

``memory_cargo``
  Custom target that runs ``cargo build``.

``ncore_memory``
  Imported static library wrapping the Cargo output.  Links against
  ``rustcsrc``, ``Threads::Threads``, ``dl``, and ``m``.

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
set(CUDA_DEVICE_MEMORY_DIR "${CMAKE_SOURCE_DIR}/ncore/native/cuda/memory")
set(HIP_DEVICE_MEMORY_DIR "${CMAKE_SOURCE_DIR}/ncore/native/hip/memory")
set(RUST_LIB "${CARGO_TARGET_DIR}/${RUST_BUILD_SUBDIR}/libncore_memory.a")

add_custom_target(memory_cargo
  COMMAND ${CMAKE_COMMAND} -E env
  "RUSTFLAGS=${RUSTFLAGS}"
  "RUST_DIR=${RUST_DIR}"
  "CUDA_DEVICE_MEMORY_DIR=${CUDA_DEVICE_MEMORY_DIR}"
  "HIP_DEVICE_MEMORY_DIR=${HIP_DEVICE_MEMORY_DIR}"
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
  Threads::Threads
  ${CMAKE_DL_LIBS}
  m
)
