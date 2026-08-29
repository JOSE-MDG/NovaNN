//! FFI-safe handle type for storage entries.
//!
//! Provides [`RustHandle`], a `repr(C)` struct that is shared across
//! the C/Rust boundary. Each handle carries a unique ID used to look
//! up the corresponding storage in the global registry, along with
//! cached metadata (size, alignment).

/// FFI-safe descriptor for one entry in the Rust storage registry.
///
/// The structure is copied by value across the C ABI. The `id` is the only
/// authoritative identity; `size_bytes` and `align` are cached metadata used
/// by the native core. The Rust registry remains the source of truth for the
/// allocation and reference count.
///
/// A handle with `id == 0` is the invalid sentinel returned after a failed
/// reservation or after the final successful release. The structure must
/// remain `#[repr(C)]` and trivially copyable because its layout is mirrored
/// by `RustHandle` in `ncore/core/storage.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustHandle {
    /// Unique key used to look up storage inside the global registry.
    pub id: u64,
    /// Cached allocation size in bytes. The authoritative value lives in
    /// [`crate::storage::RustStorage`].
    pub size_bytes: usize,
    /// Requested or recorded memory alignment in bytes.
    pub align: usize,
}

impl RustHandle {
    /// Creates a handle from a registry ID and cached allocation metadata.
    pub fn new(id: u64, size_bytes: usize, align: usize) -> Self {
        Self {
            id,
            size_bytes,
            align,
        }
    }

    /// Returns the invalid sentinel handle with `id == 0`.
    pub fn invalid() -> Self {
        Self {
            id: 0,
            size_bytes: 0,
            align: 0,
        }
    }

    /// Returns `true` when the handle has a non-zero registry ID.
    ///
    /// This is only a structural check; use
    /// [`crate::ops::query::is_valid_op`] to also verify that the ID is
    /// currently registered.
    pub fn is_valid(&self) -> bool {
        self.id != 0 && self.size_bytes != 0
    }
}
