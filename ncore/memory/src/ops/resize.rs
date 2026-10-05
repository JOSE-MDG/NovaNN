//! Storage resize operation.
//!
//! Changes the size of an allocated memory block while preserving
//! its existing contents (up to the minimum of the old and new size).
//!
//! The handle cache is updated only after the underlying storage operation
//! reports success.

use crate::error::StorageError;
use crate::handle::RustHandle;
use crate::manager::StorageManager;

/// Resizes the storage associated with the handle to `requested` bytes.
///
/// The input is the new theoretical size. Device-backed storage rounds it to
/// its size class with the same rules as the initial reservation. On success
/// the handle's cached `requested_bytes`, `bytes` and `align` fields are
/// updated to reflect the storage object.
///
/// # Arguments
///
/// * `handle`    - Mutable reference to the handle (sizes are updated in place).
/// * `requested` - Desired new theoretical size in bytes. Must be non-zero.
///
/// # Errors
///
/// Returns [`StorageError::InvalidHandle`] if the handle is not in
/// the registry, [`StorageError::InvalidSize`] if `requested` is zero or
/// cannot be rounded, [`StorageError::InvalidAlignment`] if the CPU layout
/// cannot be created, [`StorageError::ResizeFailed`] if the CPU reallocation
/// fails, or [`StorageError::DeviceError`] if the device backend rejects the
/// resize.
pub fn resize_op(handle: &mut RustHandle, requested: usize) -> Result<(), StorageError> {
    if requested == 0 {
        return Err(StorageError::InvalidSize);
    }
    let (effective_requested, effective_bytes, effective_align) =
        StorageManager::with(handle.id, |s: &mut crate::storage::RustStorage| {
            let status = s.resize(requested);
            (status, s.requested_bytes, s.size_bytes, s.align())
        })
        .map(|(status, requested_bytes, size_bytes, align)| {
            status.map(|()| (requested_bytes, size_bytes, align))
        })??;
    handle.requested_bytes = effective_requested;
    handle.bytes = effective_bytes;
    handle.align = effective_align;
    Ok(())
}
