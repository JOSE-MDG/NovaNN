//! C-compatible storage resizing.
//!
//! Exports [`resize`] so that C code can change the size of an
//! allocated memory block while preserving its contents.

use crate::error::StorageError;
use crate::handle::RustHandle;
use crate::ops::resize::resize_op;
use crate::status::NovaStatus;

/// Resizes the storage associated with the handle.
///
/// On success the handle's cached `size_bytes` is updated.
///
/// Returns a [`NovaStatus`] describing success or failure.
///
/// # Arguments
///
/// * `handle` - Mutable pointer to the handle whose allocation is resized.
/// * `new_size` - New allocation size in bytes. Must be non-zero.
///
/// # Safety
///
/// `handle` must be non-null and point to a valid [`RustHandle`].
/// The handle remains valid on a normal validation or allocation failure;
/// callers must still treat backend-specific resize failures as terminal
/// until the backend reports that the buffer is usable again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resize(handle: *mut RustHandle, new_size: usize) -> NovaStatus {
    if handle.is_null() {
        return NovaStatus::from_error(&StorageError::NullPointer);
    }
    match resize_op(unsafe { &mut *handle }, new_size) {
        Ok(()) => NovaStatus::success(),
        Err(error) => NovaStatus::from_error(&error),
    }
}
