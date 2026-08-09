//! Build script for the NovaNN Rust crate.
//!
//! This script links the pre-compiled C++ static library (`memorycsrc`)
//! produced by CMake and configures conditional compilation flags.
//!
//! # Environment variables
//!
//! | Variable         | Required | Default     | Description                              |
//! |------------------|----------|-------------|------------------------------------------|
//! | `RUSTCSRC_DIR`   | yes      | —           | Path to the compiled C++ static library. |
//! | `RUSTCSRC_NAME`  | no       | `memorycsrc`  | Name of the static library (without lib prefix). |
//!
//! The script expects Cargo to be invoked through CMake so that
//! `RUSTCSRC_DIR` is set correctly.

fn main() {
    let csrc_dir =
        std::env::var("RUSTCSRC_DIR").expect("RUSTCSRC_DIR not set - invoke cargo via CMake");

    let csrc_name = std::env::var("RUSTCSRC_NAME").unwrap_or_else(|_| "memorycsrc".to_string());

    // Link the pre-compiled C++ static library and its standard library runtime.
    println!("cargo:rustc-link-search=native={}", csrc_dir);
    println!("cargo:rustc-link-lib=static={}", csrc_name);
    println!("cargo:rustc-link-lib=stdc++");

    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rerun-if-env-changed=RUSTCSRC_DIR");
    println!("cargo:rerun-if-env-changed=RUSTCSRC_NAME");

    // C++ FFI source files — recompile when any of these change.
    println!("cargo:rerun-if-changed=csrc/ffi.cpp");
    println!("cargo:rerun-if-changed=csrc/ffi.hpp");
    println!("cargo:rerun-if-changed=csrc/admin.cpp");
    println!("cargo:rerun-if-changed=csrc/admin.hpp");
    println!("cargo:rerun-if-env-changed=CUDA_DEVICE_MEMORY_DIR");
    println!("cargo:rerun-if-env-changed=HIP_DEVICE_MEMORY_DIR");
}
