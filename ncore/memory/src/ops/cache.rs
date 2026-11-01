//! Device memory-cache operations.
//!
//! Provides [`empty_cache_op`], which asks the active GPU backend to
//! return the reclaimable backing of its memory pool. The memory that
//! live allocations hold stays reserved, so the operation never
//! invalidates a buffer.

use crate::error::StorageError;
use crate::ffi::cpp::{deviceEmptyCache, getDeviceBackend};
use crate::status::NovaError;
use std::ffi::CStr;

/// Releases the reclaimable backing of the device memory pool.
///
/// The pool keeps at least `min_bytes_to_keep` resident bytes and
/// returns the rest to the runtime.
///
/// # Errors
///
/// Returns [`StorageError::DeviceError`] when the backend rejects the
/// trim.
pub fn empty_cache_op(device_id: i32, min_bytes_to_keep: usize) -> Result<(), StorageError> {
    // SAFETY: both entry points take no pointers and the bridge owns
    // every buffer they touch.
    let kind = unsafe { getDeviceBackend() };
    let status = unsafe { deviceEmptyCache(kind, device_id, min_bytes_to_keep) };
    if status.err == NovaError::Success {
        return Ok(());
    }

    let message = if status.message.is_null() {
        "Device operation failed with no error message".to_string()
    } else {
        // SAFETY: message points to a valid C string owned by the bridge.
        unsafe { CStr::from_ptr(status.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(StorageError::DeviceError {
        code: status.err,
        message: format!("Device cache trim failed: {message}"),
    })
}
