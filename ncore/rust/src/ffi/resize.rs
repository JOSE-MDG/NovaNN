//! C-compatible storage resizing.
//!
//! Exports [`resize`] so that C code can change the size of an
//! allocated memory block while preserving its contents.

use crate::ffi::reserve::LAST_ERROR;
use crate::handle::RustHandle;
use crate::ops::resize::resize_op;

/// Resizes the storage associated with the handle.
///
/// On success the handle's cached `size_bytes` is updated.
///
/// Returns `true` on success, `false` on failure (invalid handle,
/// zero size, or allocation failure). On failure, the error message
/// can be retrieved via [`get_last_reserve_error`].
///
/// # Safety
///
/// `handle` must be non-null and point to a valid [`RustHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resize(handle: *mut RustHandle, new_size: usize) -> bool {
    if handle.is_null() {
        LAST_ERROR.with(|cell| *cell.borrow_mut() = Some("[RUST] resize: null handle".into()));
        return false;
    }
    match resize_op(unsafe { &mut *handle }, new_size) {
        Ok(()) => true,
        Err(e) => {
            let msg = format!("resize: {e}");
            LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(msg));
            false
        }
    }
}
