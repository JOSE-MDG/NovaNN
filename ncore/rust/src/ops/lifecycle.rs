//! Storage lifecycle operations.
//!
//! Manages reference counting for storage entries: [`retain_op`]
//! increments the reference count, and [`release_op`] decrements it.
//! When the count reaches zero the storage is automatically freed
//! and removed from the global registry.

use crate::error::StorageError;
use crate::handle::RustHandle;
use crate::manager::StorageManager;

/// Increments the reference count for the given handle.
///
/// Use this to extend the lifetime of a storage block that is
/// currently in use.
///
/// # Errors
///
/// Returns [`StorageError::InvalidHandle`] if the handle does not
/// exist in the registry, or [`StorageError::ManagerPoisoned`] if
/// the global mutex is poisoned.
pub fn retain_op(handle: &RustHandle) -> Result<(), StorageError> {
    StorageManager::with(handle.id, |s| s.increment_ref())
}

/// Decrements the reference count and frees storage if it reaches zero.
///
/// When the reference count drops to zero, the storage is removed
/// from the registry, which triggers [`Drop`] and deallocates the
/// underlying memory.
///
/// # Returns
///
/// `true` if the storage was freed (ref count reached zero).
///
/// # Errors
///
/// Returns [`StorageError::InvalidHandle`] if the handle does not
/// exist in the registry.
pub fn release_op(handle: &RustHandle) -> Result<bool, StorageError> {
    let should_free = StorageManager::with(handle.id, |s| s.decrement_ref())?;
    if should_free {
        StorageManager::remove(handle.id)?;
    }
    Ok(should_free)
}
