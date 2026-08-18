//! C-compatible storage lifecycle management.
//!
//! Provides [`retain`] and [`release`] for reference-counting
//! storage entries from C code.
//!
//! A successful retain or release writes a success [`NovaStatus`]. A failed
//! operation leaves ownership unchanged whenever the implementation can do
//! so, allowing the caller to report the error or retry.

use crate::error::StorageError;
use crate::handle::RustHandle;
use crate::ops::lifecycle::{release_op, retain_op};
use crate::status::NovaStatus;

/// Increments the reference count for the given handle.
///
/// # Returns
///
/// [`NovaStatus::err`](crate::NovaStatus::err) is
/// [`crate::NovaError::Success`] when the reference count was incremented;
/// otherwise the status identifies the failure.
///
/// # Safety
///
/// `handle` must be non-null and point to a valid [`RustHandle`]
/// that was previously returned by [`crate::reserve()`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn retain(handle: *mut RustHandle) -> NovaStatus {
    if handle.is_null() {
        return NovaStatus::from_error(&StorageError::NullPointer);
    }
    match retain_op(unsafe { &*handle }) {
        Ok(()) => NovaStatus::success(),
        Err(error) => NovaStatus::from_error(&error),
    }
}

/// Decrements the reference count and frees storage if it reaches zero.
///
/// Returns `true` if the storage was actually freed (reference count
/// reached zero). `status` distinguishes a live shared allocation from
/// an actual release failure.
///
/// On success with `true`, the handle is invalidated. On success with
/// `false`, another owner still holds the allocation and the handle remains
/// valid. On failure, the handle and its ownership remain available to the
/// caller, and `status` contains the failure details.
///
/// # Safety
///
/// `handle` must be non-null and point to a valid [`RustHandle`]. `status`
/// must be non-null and point to writable storage. On failure, the handle
/// remains owned by the caller and the status contains the release error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn release(handle: *mut RustHandle, status: *mut NovaStatus) -> bool {
    if status.is_null() {
        return false;
    }
    if handle.is_null() {
        unsafe {
            *status = NovaStatus::from_error(&StorageError::NullPointer);
        }
        return false;
    }
    match release_op(unsafe { &mut *handle }) {
        Ok(freed) => {
            unsafe {
                *status = NovaStatus::success();
            }
            freed
        }
        Err(error) => {
            unsafe {
                *status = NovaStatus::from_error(&error);
            }
            false
        }
    }
}
