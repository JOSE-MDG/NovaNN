//! Storage reservation operation.
//!
//! Allocates a new [`RustStorage`] block, assigns it a unique ID,
//! inserts it into the global registry, and returns an [`RustHandle`]
//! that the caller can use to reference it.

use crate::error::StorageError;
use crate::handle::RustHandle;
use crate::id::next_id;
use crate::manager::StorageManager;
use crate::storage::RustStorage;

/// Allocates a new storage block and returns a handle to it.
///
/// # Arguments
///
/// * `size`  - Number of bytes to allocate. Must be non-zero.
/// * `align` - Required memory alignment. Must be a power of two.
///
/// # Returns
///
/// A valid [`RustHandle`] on success, or a [`StorageError`] on failure
/// (invalid parameters, allocation failure, or manager poisoned).
pub fn reserve_op(size: usize, align: usize) -> Result<RustHandle, StorageError> {
    let storage = RustStorage::allocate(size, align)?;
    let id = next_id();
    StorageManager::insert(id, storage)?;
    Ok(RustHandle::new(id, size, align))
}
