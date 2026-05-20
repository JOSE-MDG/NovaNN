//! Rust memory layer for NovaNN, providing FFI-compatible storage management.

pub mod error;
pub mod ffi;
pub mod handle;
pub mod id;
pub mod manager;
pub mod ops;
pub mod storage;

pub use ffi::*;
pub use handle::RustHandle;
