//! C-compatible storage lifecycle management.
//!
//! Provides [`retain`] and [`release`] for reference-counting
//! storage entries from C code.

use crate::handle::RustHandle;
use crate::ops::lifecycle::{release_op, retain_op};

/// Increments the reference count for the given handle.
///
/// # Safety
///
/// `handle` must be non-null and point to a valid [`RustHandle`]
/// that was previously returned by [`reserve`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn retain(handle: *mut RustHandle) {
    if handle.is_null() {
        return;
    }
    let _ = retain_op(unsafe { &*handle });
}

/// Decrements the reference count and frees storage if it reaches zero.
///
/// Returns `true` if the storage was actually freed (reference count
/// reached zero).
///
/// # Safety
///
/// `handle` must be non-null and point to a valid [`RustHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn release(handle: *mut RustHandle) -> bool {
    if handle.is_null() {
        return false;
    }
    release_op(unsafe { &*handle }).unwrap_or(false)
}
