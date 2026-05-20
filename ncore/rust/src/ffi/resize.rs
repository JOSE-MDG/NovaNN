//! C-compatible storage resizing.
//!
//! Exports [`resize`] so that C code can change the size of an
//! allocated memory block while preserving its contents.

use crate::handle::RustHandle;
use crate::ops::resize::resize_op;

/// Resizes the storage associated with the handle.
///
/// On success the handle's cached `size_bytes` is updated.
///
/// Returns `true` on success, `false` on failure (invalid handle,
/// zero size, or allocation failure).
///
/// # Safety
///
/// `handle` must be non-null and point to a valid [`RustHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resize(handle: *mut RustHandle, new_size: usize) -> bool {
    if handle.is_null() {
        return false;
    }
    resize_op(unsafe { &mut *handle }, new_size).is_ok()
}
