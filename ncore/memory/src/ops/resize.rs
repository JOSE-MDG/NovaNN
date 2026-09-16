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

/// Resizes the storage associated with the handle to `new_size` bytes.
///
/// On success the handle's cached `size_bytes` and `align` fields are updated
/// to reflect the storage object.
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
/// [`StorageError::InvalidAlignment`] if the CPU layout cannot be created,
/// [`StorageError::ResizeFailed`] if the CPU reallocation fails, or
/// [`StorageError::DeviceError`] if the device backend rejects the resize.
pub fn resize_op(handle: &mut RustHandle, new_size: usize) -> Result<(), StorageError> {
    let effective_align = StorageManager::with(handle.id, |s: &mut crate::storage::RustStorage| {
        let status = s.resize(new_size);
        let align = s.align();
        (status, align)
    })
    .map(|(status, align)| status.map(|()| align))??;
    handle.size_bytes = new_size;
    handle.align = effective_align;
    Ok(())
}
