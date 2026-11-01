//! C-compatible device memory-cache management.
//!
//! Provides [`empty_cache`] for C code that wants to return the
//! reclaimable backing of the device memory pool to the runtime.

use crate::ops::cache::empty_cache_op;
use crate::status::NovaStatus;

/// Releases the reclaimable backing of the device memory pool.
///
/// The pool keeps at least `min_bytes_to_keep` resident bytes and
/// returns the rest to the runtime. Memory that live allocations hold
/// stays reserved, so the operation never invalidates a buffer.
///
/// # Arguments
///
/// * `device_id` - Runtime device ordinal owning the pool.
/// * `min_bytes_to_keep` - Minimum resident bytes the pool retains.
///
/// # Returns
///
/// [`NovaStatus::err`](crate::NovaStatus::err) is
/// [`crate::NovaError::Success`] when the pool was trimmed; otherwise the
/// status identifies the failure.
#[unsafe(no_mangle)]
pub extern "C" fn empty_cache(device_id: i32, min_bytes_to_keep: usize) -> NovaStatus {
    match empty_cache_op(device_id, min_bytes_to_keep) {
        Ok(()) => NovaStatus::success(),
        Err(error) => NovaStatus::from_error(&error),
    }
}
