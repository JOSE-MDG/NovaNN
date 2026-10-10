//! Storage reservation operation.
//!
//! Allocates a new [`RustStorage`] block (CPU RAM or GPU VRAM / pinned host),
//! assigns it a unique ID, inserts it into the global registry, and returns
//! a [`RustHandle`] that the caller can use to reference it.
//!
//! This layer does not cross an ABI boundary. It returns the native Rust
//! [`Result`] so the FFI wrapper can convert failures into [`crate::NovaStatus`]
//! values without losing context.

use crate::error::StorageError;
use crate::handle::RustHandle;
use crate::id::next_id;
use crate::manager::StorageManager;
use crate::storage::RustStorage;

/// Allocates a new storage block and returns a handle to it.
///
/// # Arguments
///
/// * `rounded`      - Reservation size in bytes, already rounded to its size
///                    class by the caller. Must be non-zero.
/// * `requested`    - Theoretical size asked by the caller. Must be non-zero
///                    and not greater than `rounded`.
/// * `device`       - Target memory device: `"cpu"` for host RAM,
///                    `"device"` for GPU VRAM.
/// * `pin_memory`   - If `true`, allocate page-locked host memory.
///                    Only valid when `device` is `"cpu"`.
/// * `align` - Required memory alignment. Must be a power of two. Ignored for device and pinned host allocations.
///
/// # Errors
///
/// Returns [`StorageError::InvalidSize`] when `rounded` or `requested` is
/// zero or when `requested` exceeds `rounded`.
/// Returns [`StorageError::InvalidDevice`] when `device` is not `"cpu"` or
/// `"device"`. Returns [`StorageError::PinnedMemoryOnDevice`] when
/// `pin_memory` is `true` and `device` is `"device"`. Propagates allocation
/// errors from the underlying backend.
///
/// # Returns
///
/// A valid [`RustHandle`] on success.
///
/// The handle owns one reference to the newly inserted registry entry. Its
/// cached sizes are taken from the actual storage object.
pub fn reserve_op(
    rounded: usize,
    requested: usize,
    device: &str,
    pin_memory: bool,
    align: usize,
) -> Result<RustHandle, StorageError> {
    if rounded == 0 || requested == 0 || requested > rounded {
        return Err(StorageError::InvalidSize);
    }
    let storage = match device {
        "cpu" if pin_memory => RustStorage::allocate_device(rounded, requested, true)?,
        "cpu" => RustStorage::allocate(rounded, requested, align)?,
        "device" if pin_memory => return Err(StorageError::PinnedMemoryOnDevice),
        "device" => RustStorage::allocate_device(rounded, requested, false)?,
        _ => return Err(StorageError::InvalidDevice),
    };

    let id = next_id();
    let requested_bytes = storage.requested_bytes;
    let size_bytes = storage.size_bytes;
    let effective_align = storage.align();
    StorageManager::insert(id, storage)?;
    Ok(RustHandle::new(
        id,
        requested_bytes,
        size_bytes,
        effective_align,
    ))
}
