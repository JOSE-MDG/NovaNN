//! C-compatible storage queries.
//!
//! Provides [`get_data_from`] for accessing the raw memory pointer,
//! [`is_valid_handle`] for checking handle validity, and metadata queries
//! for allocation alignment, device-backed memory, and pinned host memory.

use crate::handle::RustHandle;
use crate::ops::query::{
    get_align_op, get_data_op, is_device_memory_op, is_pinned_op, is_valid_op,
};
use std::cell::RefCell;
use std::ffi::{CString, c_char, c_int, c_void};

thread_local! {
    /// Thread-local storage for the most recent allocation error message.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Stores the most recent allocation failure message for the current thread.
///
/// The message is converted to a NUL-terminated [`CString`] so that
/// [`get_last_reserve_error`] can hand a safe `*const c_char` to C code.
/// Messages produced by [`crate::error::StorageError`] never contain
/// interior NUL bytes; the fallback only guards against future regressions.
pub(crate) fn set_last_error(msg: impl Into<String>) {
    let cmsg = CString::new(msg.into()).unwrap_or_else(|_| {
        CString::new("storage error message contained an interior NUL byte")
            .expect("static fallback is NUL-free")
    });
    LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(cmsg));
}

/// Returns the last allocation error message, or a null pointer if the last
/// reserve or resize call succeeded.
///
/// The returned pointer is valid until the next `reserve()` or `resize()`
/// call on the same thread.
///
/// # Safety
///
/// The returned pointer must not be freed or modified by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_last_reserve_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some(msg) => msg.as_ptr() as *const c_char,
            None => std::ptr::null(),
        }
    })
}

/// Returns the length of the last error message, or 0 if none.
///
/// Useful for callers that want to pre-allocate a buffer or log the
/// length before copying. The length excludes the NUL terminator.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_last_reserve_error_len() -> c_int {
    LAST_ERROR.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some(msg) => msg.as_bytes().len() as c_int,
            None => 0,
        }
    })
}

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
