//! Rust memory layer for NovaNN, providing FFI-compatible storage management.
//!
//! This crate is the owner of every buffer exposed by the native core. C and
//! C++ callers receive a [`RustHandle`] and use the exported lifecycle
//! functions to reserve, share, resize, query, and release the allocation.
//!
//! # Components
//!
//! * [`storage`] owns CPU allocations and device-buffer descriptors returned
//!   by the C++ backend.
//! * [`counter`] provides atomic reference-count storage for live owners.
//! * [`manager`] stores active allocations in a process-wide registry.
//! * [`ops`] implements fallible storage operations over the registry.
//! * [`ffi`] exposes the operations through the C ABI.
//! * [`status`] defines the ABI-compatible error codes and status values.
//!
//! # Allocation and lifecycle
//!
//! A successful [`reserve()`] call creates one reference. Each successful
//! [`retain()`] call adds an owner, and each successful [`release()`] call
//! removes one. The final release deallocates the buffer before removing its
//! entry from the registry. If deallocation fails, the reference and registry
//! entry remain available for recovery.
//!
//! CPU allocations use the system allocator. GPU and pinned-host allocations
//! use the C++ device-memory backend. The allocation backend is retained with
//! each [`storage::RustStorage`] so resize and release use the corresponding
//! backend operations.
//!
//! # Error reporting
//!
//! Fallible Rust operations return [`Result`]. Exported FFI functions convert
//! those results into [`NovaStatus`] values containing a [`NovaError`] code
//! and a NUL-terminated diagnostic message. A status message remains valid
//! until the next storage operation on the same thread.
//!
//! # Synchronization
//!
//! The storage registry serializes access to allocation state with a mutex.
//! Reference counts use [`counter::AtomicRefCounter`], while handle values
//! remain caller-owned and must not be concurrently mutated without external
//! synchronization.

#![warn(missing_docs)]

pub mod counter;
pub mod error;
pub mod ffi;
pub mod handle;
pub mod id;
pub mod manager;
pub mod ops;
pub mod status;
pub mod storage;

pub use ffi::*;
pub use handle::RustHandle;
pub use status::{NovaError, NovaStatus};
