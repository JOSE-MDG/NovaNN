//! Storage reservation operation.
//!
//! Allocates a new [`RustStorage`] block (CPU RAM or GPU VRAM / pinned host),
//! assigns it a unique ID, inserts it into the global registry, and returns
//! a [`RustHandle`] that the caller can use to reference it.

use crate::error::StorageError;
use crate::handle::RustHandle;
use crate::id::next_id;
use crate::manager::StorageManager;
use crate::storage::RustStorage;

/// Allocates a new storage block and returns a handle to it.
///
/// # Arguments
///
/// * `size`       - Number of bytes to allocate. Must be non-zero.
/// * `device`     - Target memory device: `"cpu"` for host RAM,
///                  `"device"` for GPU VRAM.
/// * `pin_memory` - If `true`, allocate page-locked host memory.
///                  Only valid when `device` is `"cpu"`.
/// * `align`      - Required memory alignment. Must be a power of two.
///
/// # Errors
///
/// Returns [`StorageError::InvalidDevice`] when `device` is not `"cpu"` or
/// `"device"`. Returns [`StorageError::PinnedMemoryOnDevice`] when
/// `pin_memory` is `true` and `device` is `"device"`. Propagates allocation
/// errors from the underlying backend.
///
/// # Returns
///
/// A valid [`RustHandle`] on success.
pub fn reserve_op(
    size: usize,
    device: &str,
    pin_memory: bool,
    align: usize,
) -> Result<RustHandle, StorageError> {
    let storage = match device {
        "cpu" if pin_memory => RustStorage::allocate_device(size, pin_memory)?,
        "cpu" => RustStorage::allocate(size, align)?,
        "device" if pin_memory => return Err(StorageError::PinnedMemoryOnDevice),
        "device" => RustStorage::allocate_device(size, false)?,
        _ => return Err(StorageError::InvalidDevice),
    };

    let id = next_id();
    let size_bytes = storage.size_bytes;
    StorageManager::insert(id, storage)?;
    Ok(RustHandle::new(id, size_bytes, align))
}
