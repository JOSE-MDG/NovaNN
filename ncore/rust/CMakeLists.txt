#[[
    @file ncore/rust/CMakeLists.txt
    @brief Rust / C++ FFI bridge build orchestration.

    Pipeline:
      1. add_subdirectory(csrc) compiles the C++ FFI + device back-ends into
         the static library librustcsrc.a.
      2. An add_custom_target (ncore_memory_cargo) invokes Cargo, passing:
         - RUSTCSRC_DIR — location of librustcsrc.a (CMake generator expression)
         - RUSTCSRC_NAME — library name ("rustcsrc")
         - RUSTFLAGS — tuned per build type (debuginfo / optimisation)
         - CARGO_PROFILE — empty (debug) or --release
      3. Cargo produces libncore_memory.a (a staticlib that links rustcsrc).
      4. An IMPORTED STATIC target ncore_memory wraps the .a and exposes
         rustcsrc, Threads::Threads, dl, and m as INTERFACE link requirements.

    CMake → Cargo dependency: ncore_memory_cargo DEPENDS rustcsrc, so the
    C++ static library is always up-to-date before Cargo links against it.

    @note If CMAKE_BUILD_TYPE is unset, the Release branch is taken.
    @note build.rs reads RUSTCSRC_DIR and RUSTCSRC_NAME from the environment.

    Requires : CMake 3.27+, Cargo, Rust toolchain
    Depends  : ncore/rust/csrc/ (librustcsrc.a producer)
    Outputs  : ncore_memory (imported static library target)
    See also : build.rs — Rust-side linkage of librustcsrc.a
]]

add_subdirectory(csrc)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RUSTFLAGS "-C debuginfo=2 -C opt-level=0")
    set(CARGO_PROFILE "")
    set(RUST_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/debug")
else()
    set(RUSTFLAGS "-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C panic=abort")
    set(CARGO_PROFILE "--release")
    set(RUST_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/release")
endif()

set(RUST_LIB "${RUST_LIB_DIR}/libncore_memory.a")

add_custom_target(ncore_memory_cargo
    COMMAND ${CMAKE_COMMAND} -E env
    "RUSTFLAGS=${RUSTFLAGS}"
    "RUSTCSRC_DIR=$<TARGET_FILE_DIR:rustcsrc>"
    "RUSTCSRC_NAME=rustcsrc"
    "NOVA_HAS_CUDA=${NOVA_HAS_CUDA}"
    "NOVA_HAS_HIP=${NOVA_HAS_HIP}"
    cargo build ${CARGO_PROFILE}
    --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
    BYPRODUCTS ${RUST_LIB}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building Rust crate ncore_memory (${CMAKE_BUILD_TYPE})"
    DEPENDS rustcsrc
)

add_library(ncore_memory STATIC IMPORTED GLOBAL)
set_target_properties(ncore_memory PROPERTIES
    IMPORTED_LOCATION ${RUST_LIB}
)
add_dependencies(ncore_memory ncore_memory_cargo)

target_link_libraries(ncore_memory INTERFACE
    rustcsrc
    Threads::Threads
    ${CMAKE_DL_LIBS}
    m
)
