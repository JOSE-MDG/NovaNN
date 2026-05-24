//! Storage resize operation.
//!
//! Changes the size of an allocated memory block while preserving
//! its existing contents (up to the minimum of the old and new size).

use crate::error::StorageError;
use crate::handle::RustHandle;
use crate::manager::StorageManager;

/// Resizes the storage associated with the handle to `new_size` bytes.
///
/// On success the handle's cached `size_bytes` field is updated to
/// reflect the new size.
///
/// # Arguments
///
/// * `handle`   - Mutable reference to the handle (size is updated in place).
/// * `new_size` - Desired new size in bytes. Must be non-zero.
///
/// # Errors
///
/// Returns [`StorageError::InvalidHandle`] if the handle is not in
/// the registry, [`StorageError::InvalidSize`] if `new_size` is zero,
/// or [`StorageError::ResizeFailed`] if the reallocation fails.
pub fn resize_op(handle: &mut RustHandle, new_size: usize) -> Result<(), StorageError> {
    StorageManager::with(handle.id, |s: &mut crate::storage::RustStorage| {
        s.resize(new_size)
    })??;
    handle.size_bytes = new_size;
    Ok(())
}
