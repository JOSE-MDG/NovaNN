//! C-compatible storage reservation.
//!
//! Exports [`reserve`] so that C code can allocate storage on a chosen
//! device (CPU RAM, GPU VRAM, or pinned host memory) and obtain a handle.
//!
//! The function reports all failures through its caller-provided
//! [`NovaStatus`] pointer. It never panics across the C ABI.

use crate::error::StorageError;
use crate::handle::RustHandle;
use crate::ops::reserve::reserve_op;
use crate::status::{NovaError, NovaStatus};
use std::ffi::{CStr, c_char};

/// Allocates storage on the specified device.
///
/// # Arguments
///
/// * `size`       - Number of bytes to allocate. Must be non-zero.
/// * `device`     - C string: `"cpu"` for host RAM, `"device"` for GPU VRAM.
///                  A null pointer is treated as `"cpu"` for compatibility
///                  with the native allocator wrapper.
/// * `pin_memory` - If `true`, allocate page-locked host memory (only valid
///                  when `device` is `"cpu"`).
/// * `align`      - Required memory alignment. Must be a power of two.
/// * `status`     - Non-null output status populated for every call.
///
/// # Returns
///
/// A valid [`RustHandle`] on success. Returns an invalid handle (ID = 0)
/// on failure and writes the failure to `status`.
/// On success, `status.err` is [`NovaError::Success`].
///
/// # Safety
///
/// `device` must be null or point to a valid null-terminated C string.
/// `status` must be non-null and point to writable storage. The returned
/// [`RustHandle`] must be passed to [`crate::release()`] when the caller is done with
/// the storage to avoid memory leaks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reserve(
    size: usize,
    device: *const c_char,
    pin_memory: bool,
    align: usize,
    status: *mut NovaStatus,
) -> RustHandle {
    if status.is_null() {
        return RustHandle::invalid();
    }
    unsafe {
        *status = NovaStatus::success();
    }

    let dev = if device.is_null() {
        "cpu"
    } else {
        // SAFETY: caller guarantees a valid C string.
        match unsafe { CStr::from_ptr(device) }.to_str() {
            Ok(s) => s,
            Err(error) => {
                let error = StorageError::DeviceError {
                    code: NovaError::InvalidDevice,
                    message: format!("Device string is not valid UTF-8: {error}"),
                };
                unsafe {
                    *status = NovaStatus::from_error(&error);
                }
                return RustHandle::invalid();
            }
        }
    };
    match reserve_op(size, dev, pin_memory, align) {
        Ok(handle) => handle,
        Err(e) => {
            unsafe {
                *status = NovaStatus::from_error(&e);
            }
            RustHandle::invalid()
        }
    }
}
