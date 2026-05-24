//! C-compatible storage reservation.
//!
//! Exports [`reserve`] so that C code can allocate storage on a chosen
//! device (CPU RAM, GPU VRAM, or pinned host memory) and obtain a handle.

use crate::handle::RustHandle;
use crate::ops::reserve::reserve_op;
use std::ffi::{CStr, c_char};

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
            Err(_) => return RustHandle::invalid(),
        }
    };
    reserve_op(size, dev, pin_memory, align).unwrap_or_else(|_| RustHandle::invalid())
}
