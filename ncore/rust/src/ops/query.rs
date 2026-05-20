//! Storage query operations.
//!
//! Provides read-only access to storage metadata and data pointers:
//! [`get_data_op`] returns a raw pointer to the allocated memory,
//! and [`is_valid_op`] checks whether a handle is currently registered.

use crate::error::StorageError;
use crate::handle::RustHandle;
use crate::manager::StorageManager;

/// Returns a raw pointer to the data for the given handle.
///
/// The pointer remains valid as long as the handle's reference count
/// is greater than zero.
///
/// # Errors
///
/// Returns [`StorageError::InvalidHandle`] if the handle does not
/// exist in the registry.
pub fn get_data_op(handle: &RustHandle) -> Result<*mut u8, StorageError> {
    StorageManager::with(handle.id, |s| s.data_ptr())
}

/// Returns `true` if the handle is both structurally valid
/// (non-zero ID) and currently registered in the storage manager.
pub fn is_valid_op(handle: &RustHandle) -> bool {
    handle.is_valid() && StorageManager::contains(handle.id)
}
