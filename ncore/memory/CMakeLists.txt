#[=======================================================================[.rst:
Rust Crate Build
----------------

Build the ``ncore_memory`` Rust crate via Cargo and expose it as an
imported static library.  The Cargo build is orchestrated through a
custom target that passes all required environment variables to
``cargo build``.

Variables defined:

``RUST_LIB``
  Path to the built static library (``libncore_memory.a``/``libncore_memory.lib``).

``CARGO_TARGET_DIR``
  Cargo output directory inside the CMake build tree.

Targets created:

``memory_cargo``
  Custom target that runs ``cargo build``.

``ncore_memory`` — alias ``ncore::memory``
  Imported static library wrapping the Cargo output.  Links against
  ``ncore::memory::csrc`` (alias of ``memorycsrc``), ``Threads::Threads``,
  ``dl``, and ``m``.

#]=======================================================================]

add_subdirectory(csrc)

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

if(WIN32)
  set(RUST_STATIC_PREFIX "")
  set(RUST_STATIC_EXT ".lib")
else()
  set(RUST_STATIC_PREFIX "lib")
  set(RUST_STATIC_EXT ".a")
endif()

set(RUST_LIB "${CARGO_TARGET_DIR}/${RUST_BUILD_SUBDIR}/${RUST_STATIC_PREFIX}ncore_memory${RUST_STATIC_EXT}")

add_custom_target(memory_cargo
  COMMAND ${CMAKE_COMMAND} -E env
  "RUSTFLAGS=${RUSTFLAGS}"
  "RUST_DIR=${RUST_DIR}"
  "CUDA_DEVICE_MEMORY_DIR=${CUDA_DEVICE_MEMORY_DIR}"
  "HIP_DEVICE_MEMORY_DIR=${HIP_DEVICE_MEMORY_DIR}"
  "RUSTCSRC_DIR=$<TARGET_FILE_DIR:ncore::memory::csrc>"
  "RUSTCSRC_NAME=memorycsrc"
  "NOVA_HAS_CUDA=${NOVA_HAS_CUDA}"
  "NOVA_HAS_HIP=${NOVA_HAS_HIP}"
  "CARGO_TARGET_DIR=${CARGO_TARGET_DIR}"
  cargo build ${CARGO_PROFILE}
  --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
  BYPRODUCTS ${RUST_LIB}
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Building Rust crate ncore_memory"
  DEPENDS ncore::memory::csrc
)

add_library(ncore_memory STATIC IMPORTED GLOBAL)
add_library(ncore::memory ALIAS ncore_memory)
set_target_properties(ncore_memory PROPERTIES
  IMPORTED_LOCATION ${RUST_LIB}
  POSITION_INDEPENDENT_CODE ON
)
add_dependencies(ncore_memory memory_cargo)

target_link_libraries(ncore_memory INTERFACE
  $<$<NOT:$<PLATFORM_ID:Windows>>:Threads::Threads>
  $<$<NOT:$<PLATFORM_ID:Windows>>:${CMAKE_DL_LIBS}>
  $<$<NOT:$<PLATFORM_ID:Windows>>:m>
)
