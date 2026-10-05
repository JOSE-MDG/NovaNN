//! FFI-safe handle type for storage entries.
//!
//! Provides [`RustHandle`], a `repr(C)` struct that is shared across
//! the C/Rust boundary. Each handle carries a unique ID used to look
//! up the corresponding storage in the global registry, along with
//! cached metadata (theoretical size, real reservation size, alignment).

/// FFI-safe descriptor for one entry in the Rust storage registry.
///
/// The structure is copied by value across the C ABI. The `id` is the only
/// authoritative identity. `requested_bytes`, `bytes` and `align` are cached
/// metadata used by the native core. `requested_bytes` is the theoretical
/// size asked by the caller while `bytes` is the real backend reservation,
/// which may be larger due to size-class rounding. The Rust registry remains
/// the source of truth for the allocation and reference count.
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
    /// Theoretical size asked by the caller, in bytes.
    pub requested_bytes: usize,
    /// Real backend reservation in bytes. Greater than or equal to [`Self::requested_bytes`].
    pub bytes: usize,
    /// Requested or recorded memory alignment in bytes.
    pub align: usize,
}

impl RustHandle {
    /// Creates a handle from a registry ID and cached allocation metadata.
    pub fn new(id: u64, requested_bytes: usize, bytes: usize, align: usize) -> Self {
        Self {
            id,
            requested_bytes,
            bytes,
            align,
        }
    }

    /// Returns the invalid sentinel handle with `id == 0`.
    pub fn invalid() -> Self {
        Self {
            id: 0,
            requested_bytes: 0,
            bytes: 0,
            align: 0,
        }
    }

    /// Returns `true` when the handle has a non-zero registry ID.
    ///
    /// This is only a structural check. Use
    /// [`crate::ops::query::is_valid_op`] to also verify that the ID is
    /// currently registered.
    pub fn is_valid(&self) -> bool {
        self.id != 0 && self.bytes != 0
    }
}
