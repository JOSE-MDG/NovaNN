//! Storage lifecycle operations.
//!
//! Manages reference counting for storage entries: [`retain_op`]
//! increments the reference count, and [`release_op`] decrements it.
//! When the count reaches zero the storage is explicitly deallocated and
//! then removed from the global registry.
//!
//! The operations are intentionally separate from the C ABI wrappers so the
//! registry and allocation logic can use ordinary Rust [`Result`] values.

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
/// the global mutex is poisoned. It can also return
/// [`StorageError::ReferenceCountOverflow`] when the count cannot be
/// incremented.
pub fn retain_op(handle: &RustHandle) -> Result<(), StorageError> {
    StorageManager::with(handle.id, |s| s.increment_ref())?
}

/// Decrements the reference count and frees storage if it reaches zero.
///
/// When the reference count drops to zero, the underlying memory is
/// explicitly deallocated before the storage is removed from the registry.
/// [`Drop`] remains a fallback for cleanup paths outside normal release.
///
/// # Returns
///
/// `true` if the storage was freed (ref count reached zero).
///
/// # Errors
///
/// Returns [`StorageError::InvalidHandle`] if the handle does not
/// exist in the registry, [`StorageError::ManagerPoisoned`] if the registry
/// mutex is poisoned, or a backend error if final deallocation fails. When an
/// error is returned, the handle remains valid and the reference ownership is
/// retained for recovery or retry.
pub fn release_op(handle: &mut RustHandle) -> Result<bool, StorageError> {
    let should_free: bool = StorageManager::release(handle.id)?;
    if should_free {
        *handle = RustHandle::invalid();
    }
    Ok(should_free)
}
