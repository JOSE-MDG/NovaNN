//! C-compatible storage queries.
//!
//! Provides [`get_data_from`] for accessing the raw memory pointer,
//! [`is_valid_handle`] for checking handle validity, and metadata queries
//! for allocation alignment, device-backed memory, and pinned host memory.

use crate::handle::RustHandle;
use crate::ops::query::{
    get_align_op, get_data_op, is_device_memory_op, is_pinned_op, is_valid_op,
};
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

/// Returns the allocation alignment recorded for the given handle.
///
/// Returns `0` on failure (null handle or invalid handle), which cannot be a
/// valid allocation alignment in this storage layer.
///
/// # Safety
///
/// `handle` must be null or point to a (possibly invalid)
/// [`RustHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_align_from(handle: *mut RustHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    get_align_op(unsafe { &*handle }).unwrap_or(0)
}

/// Returns `true` if the handle refers to device-backed storage.
///
/// Pinned host allocations are also device-backed because they are allocated
/// through the active GPU backend. Returns `false` for null or invalid handles.
///
/// # Safety
///
/// `handle` must be null or point to a (possibly invalid)
/// [`RustHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn is_device_memory_handle(handle: *mut RustHandle) -> bool {
    if handle.is_null() {
        return false;
    }
    is_device_memory_op(unsafe { &*handle }).unwrap_or(false)
}

/// Returns `true` if the handle refers to pinned host memory.
///
/// Returns `false` for CPU system allocations, null handles, and invalid
/// handles.
///
/// # Safety
///
/// `handle` must be null or point to a (possibly invalid)
/// [`RustHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn is_pinned_handle(handle: *mut RustHandle) -> bool {
    if handle.is_null() {
        return false;
    }
    is_pinned_op(unsafe { &*handle }).unwrap_or(false)
}
