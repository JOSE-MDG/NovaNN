//! build.rs — Cargo build script for the ncore_memory crate.
//!
//! # Role
//!
//! Instructs rustc to link the C++ FFI static library (`librustcsrc.a`) so that
//! the Rust crate can call `extern "C"` functions declared in
//! `src/ffi/cpp/bindings.rs` (device_reserve, device_release, device_memcpy,
//! get_device_backend, etc.).
//!
//! # Environment Variables (set by CMake)
//!
//! | Variable        | Description                                        |
//! |-----------------|----------------------------------------------------|
//! | `RUSTCSRC_DIR`  | Directory containing `librustcsrc.a`               |
//! | `RUSTCSRC_NAME` | Library name without prefix/extension (default: `rustcsrc`) |
//!
//! # Rerun Triggers
//!
//! Cargo watches `build.rs`, the `RUSTCSRC_DIR` env-var, and every C++ source
//! and header under `csrc/`.  Any change triggers a rebuild of `libncore_memory.a`.
//!
//! # Linking
//!
//! - `librustcsrc.a` (static) — the C++ FFI and device back-ends.
//! - `libstdc++.a` (static)   — C++ standard library, resolved via `-lstdc++`.

fn main() {
    let csrc_dir =
        std::env::var("RUSTCSRC_DIR").expect("RUSTCSRC_DIR not set — invoke cargo via CMake");

    let csrc_name = std::env::var("RUSTCSRC_NAME").unwrap_or_else(|_| "rustcsrc".to_string());

    println!("cargo:rustc-link-search=native={csrc_dir}");
    println!("cargo:rustc-link-lib=static={csrc_name}");

    println!("cargo:rustc-link-lib=stdc++");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTCSRC_DIR");

    // FFI layer
    println!("cargo:rerun-if-changed=csrc/ffi.cpp");
    println!("cargo:rerun-if-changed=csrc/ffi.hpp");

    // Device admin
    println!("cargo:rerun-if-changed=csrc/device/admin.cpp");
    println!("cargo:rerun-if-changed=csrc/device/admin.hpp");

    // CUDA backend
    println!("cargo:rerun-if-changed=csrc/device/cuda/cuda_allocator.cpp");
    println!("cargo:rerun-if-changed=csrc/device/cuda/cuda_allocator.hpp");
    println!("cargo:rerun-if-changed=csrc/device/cuda/cuda_io.cpp");
    println!("cargo:rerun-if-changed=csrc/device/cuda/cuda_io.hpp");

    // HIP backend
    println!("cargo:rerun-if-changed=csrc/device/hip/hip_allocator.cpp");
    println!("cargo:rerun-if-changed=csrc/device/hip/hip_allocator.hpp");
    println!("cargo:rerun-if-changed=csrc/device/hip/hip_io.cpp");
    println!("cargo:rerun-if-changed=csrc/device/hip/hip_io.hpp");
}
