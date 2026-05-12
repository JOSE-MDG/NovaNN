//! High-level storage operations.

use crate::error::StorageError;
use crate::handle::{next_id, RustHandle};
use crate::manager::StorageManager;
use crate::storage::RustStorage;

/// Allocates new storage and returns a handle.
pub fn reserve_op(size: usize, align: usize) -> Result<RustHandle, StorageError> {
    let storage = RustStorage::allocate(size, align)?;
    let id = next_id();
    StorageManager::insert(id, storage)?;
    Ok(RustHandle::new(id, size, align))
}

/// Increments the reference count for the given handle.
pub fn retain_op(handle: &RustHandle) -> Result<(), StorageError> {
    StorageManager::with(handle.id, |s| s.increment_ref())
}

/// Decrements the reference count and frees storage if it reaches zero.
pub fn release_op(handle: &RustHandle) -> Result<bool, StorageError> {
    let should_free = StorageManager::with(handle.id, |s| s.decrement_ref())?;
    if should_free {
        StorageManager::remove(handle.id)?;
    }
    Ok(should_free)
}

/// Resizes the storage associated with the handle.
pub fn resize_op(handle: &mut RustHandle, new_size: usize) -> Result<(), StorageError> {
    StorageManager::with(handle.id, |s| s.resize(new_size))??;
    handle.size_bytes = new_size;
    Ok(())
}

/// Returns a pointer to the data for the given handle.
pub fn get_data_op(handle: &RustHandle) -> Result<*mut u8, StorageError> {
    StorageManager::with(handle.id, |s| s.data_ptr())
}

/// Returns `true` if the handle is valid and exists in the registry.
pub fn is_valid_op(handle: &RustHandle) -> bool {
    handle.is_valid() && StorageManager::contains(handle.id)
}
