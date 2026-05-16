#[[
    @file ncore/rust/CMakeLists.txt
    @brief NovaNN Rust memory library build configuration.
    @details Builds libncore_memory.a via Cargo, applies build-type
             specific flags (native optimization for Release, debug for Debug).

    Requires: Cargo, Rust toolchain
    Output: libncore_memory.a
]]
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RUSTFLAGS "-C -g -C opt-level=0")
    set(CARGO_PROFILE "")
    set(RUST_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/debug")
else()
    set(RUSTFLAGS "-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C panic=abort")
    set(CARGO_PROFILE "--release")
    set(RUST_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/release")
endif()

set(RUST_LIB "${RUST_LIB_DIR}/libncore_memory.a")

add_custom_target(ncore_memory_cargo
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=${RUSTFLAGS}"
    cargo build ${CARGO_PROFILE} --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
    BYPRODUCTS ${RUST_LIB}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building Rust crate ncore_memory in ${CMAKE_BUILD_TYPE} mode"
)

# ncore memory library
add_library(ncore_memory STATIC IMPORTED GLOBAL)
set_target_properties(ncore_memory PROPERTIES
    IMPORTED_LOCATION ${RUST_LIB}
)
add_dependencies(ncore_memory ncore_memory_cargo)

# find Threads package and link
find_package(Threads REQUIRED)

target_link_libraries(ncore_memory INTERFACE
    Threads::Threads
    ${CMAKE_DL_LIBS}
    m
)
