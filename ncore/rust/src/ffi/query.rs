//! C-compatible storage queries.
//!
//! Provides [`get_data_from`] for accessing the raw memory pointer
//! and [`is_valid_handle`] for checking handle validity, both
//! callable from C.

use crate::handle::RustHandle;
use crate::ops::query::{get_data_op, is_valid_op};
use std::ffi::c_void;

/// Returns a pointer to the data for the given handle.
///
/// Returns null on failure (null handle or invalid handle).
///
/// # Safety
///
/// `handle` must be non-null and point to a valid [`RustHandle`],
/// or be safely reconstructible from a null check.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_data_from(handle: *mut RustHandle) -> *mut c_void {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    get_data_op(unsafe { &*handle }).unwrap_or(std::ptr::null_mut()) as *mut c_void
}

/// Returns `true` if the handle is structurally valid and
/// currently registered in the storage manager.
///
/// # Safety
///
/// `handle` must be null or point to a (possibly invalid)
/// [`RustHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn is_valid_handle(handle: *mut RustHandle) -> bool {
    if handle.is_null() {
        return false;
    }
    is_valid_op(unsafe { &*handle })
}
