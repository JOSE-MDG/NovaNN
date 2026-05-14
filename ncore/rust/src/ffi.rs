//! FFI bindings for the storage system.
//!
//! These functions are C-compatible and can be called from other languages.

use crate::handle::RustHandle;
use crate::ops;
use std::ffi::c_void;

/// Allocates new storage with the given size and alignment.
/// Returns an invalid handle on failure.
#[unsafe(no_mangle)]
pub extern "C" fn reserve(size: usize, align: usize) -> RustHandle {
    ops::reserve_op(size, align).unwrap_or_else(|_| RustHandle::invalid())
}

/// Increments the reference count for the given handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn retain(handle: *mut RustHandle) {
    if handle.is_null() {
        return;
    }
    let _ = ops::retain_op(unsafe { &*handle });
}

/// Decrements the reference count and frees storage if it reaches zero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn release(handle: *mut RustHandle) -> bool {
    if handle.is_null() {
        return false;
    }
    ops::release_op(unsafe { &*handle }).unwrap_or(false)
}

/// Resizes the storage associated with the handle.
/// Returns `true` on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resize(handle: *mut RustHandle, new_size: usize) -> bool {
    if handle.is_null() {
        return false;
    }
    ops::resize_op(unsafe { &mut *handle }, new_size).is_ok()
}

/// Returns a pointer to the data for the given handle.
/// Returns null on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_data_from(handle: *mut RustHandle) -> *mut c_void {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    ops::get_data_op(unsafe { &*handle }).unwrap_or(std::ptr::null_mut()) as *mut c_void
}

/// Returns `true` if the handle is valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn is_valid_handle(handle: *mut RustHandle) -> bool {
    if handle.is_null() {
        return false;
    }
    ops::is_valid_op(unsafe { &*handle })
}
