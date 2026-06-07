//! C-compatible storage reservation.
//!
//! Exports [`reserve`] so that C code can allocate storage on a chosen
//! device (CPU RAM, GPU VRAM, or pinned host memory) and obtain a handle.
//! On failure, the error message is stored in a thread-local buffer and
//! can be retrieved via [`get_last_reserve_error`].

use crate::handle::RustHandle;
use crate::ops::reserve::reserve_op;
use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_int};

thread_local! {
    pub(crate) static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Returns the last error message from [`reserve`], or a null pointer if
/// the last call succeeded.
///
/// The returned pointer is valid until the next call to [`reserve`] on
/// the same thread.
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
/// length before copying.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_last_reserve_error_len() -> c_int {
    LAST_ERROR.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some(msg) => msg.len() as c_int,
            None => 0,
        }
    })
}

/// Allocates storage on the specified device.
///
/// # Parameters
///
/// * `size`       - Number of bytes to allocate. Must be non-zero.
/// * `device`     - C string: `"cpu"` for host RAM, `"device"` for GPU VRAM.
/// * `pin_memory` - If `true`, allocate page-locked host memory (only valid
///                  when `device` is `"cpu"`).
/// * `align`      - Required memory alignment. Must be a power of two.
///
/// # Returns
///
/// A valid [`RustHandle`] on success. Returns an invalid handle (ID = 0)
/// on failure so that callers can check success with [`is_valid_handle`].
///
/// # Safety
///
/// `device` must be a valid null-terminated C string. The returned
/// [`RustHandle`] must be passed to [`release`] when the caller is
/// done with the storage to avoid memory leaks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reserve(
    size: usize,
    device: *const c_char,
    pin_memory: bool,
    align: usize,
) -> RustHandle {
    let dev = if device.is_null() {
        "cpu"
    } else {
        // SAFETY: caller guarantees a valid C string.
        match unsafe { CStr::from_ptr(device) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                let msg = format!("reserve: invalid device string: {e}");
                LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(msg));
                return RustHandle::invalid();
            }
        }
    };
    match reserve_op(size, dev, pin_memory, align) {
        Ok(handle) => handle,
        Err(e) => {
            let msg = format!("reserve: {e}");
            LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(msg));
            RustHandle::invalid()
        }
    }
}
