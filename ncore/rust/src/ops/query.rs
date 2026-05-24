//! Storage query operations.
//!
//! Provides read-only access to storage metadata and data pointers:
//! [`get_data_op`] returns a raw pointer to the allocated memory,
//! [`is_valid_op`] checks whether a handle is currently registered,
//! [`get_align_op`] reports the allocation alignment, and
//! [`is_device_memory_op`] / [`is_pinned_op`] expose backend placement flags.

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

/// Returns the alignment used by the storage referenced by `handle`.
///
/// CPU storage reports the [`std::alloc::Layout`] alignment used for the
/// allocation. Device-backed storage reports the alignment requested through
/// the device allocation path.
///
/// # Errors
///
/// Returns [`StorageError::InvalidHandle`] if the handle is not registered.
pub fn get_align_op(handle: &RustHandle) -> Result<usize, StorageError> {
    StorageManager::with(handle.id, |s| s.align())
}

/// Returns `true` when the handle points to device-backed storage.
///
/// This is `true` for GPU allocations managed through the C++ device FFI,
/// including pinned host allocations, and `false` for storage allocated with
/// the CPU system allocator.
///
/// # Errors
///
/// Returns [`StorageError::InvalidHandle`] if the handle is not registered.
pub fn is_device_memory_op(handle: &RustHandle) -> Result<bool, StorageError> {
    StorageManager::with(handle.id, |s| s.is_device_memory())
}

/// Returns `true` when the handle points to pinned host memory.
///
/// CPU system allocations always return `false`. Device-backed storage returns
/// the pinned flag recorded by the backend allocation descriptor.
///
/// # Errors
///
/// Returns [`StorageError::InvalidHandle`] if the handle is not registered.
pub fn is_pinned_op(handle: &RustHandle) -> Result<bool, StorageError> {
    StorageManager::with(handle.id, |s| s.is_pinned())
}
