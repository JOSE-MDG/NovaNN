use crate::error::StorageError;
use crate::handle::{next_id, RustHandle};
use crate::manager::StorageManager;
use crate::storage::RustStorage;

pub fn reserve_op(size: usize, align: usize) -> Result<RustHandle, StorageError> {
    let storage = RustStorage::allocate(size, align)?;
    let id = next_id();
    StorageManager::insert(id, storage)?;
    Ok(RustHandle::new(id, size, align))
}

pub fn retain_op(handle: &RustHandle) -> Result<(), StorageError> {
    StorageManager::with(handle.id, |s| s.increment_ref())
}

pub fn release_op(handle: &RustHandle) -> Result<(), StorageError> {
    let should_free = StorageManager::with(handle.id, |s| s.decrement_ref())?;
    if should_free {
        StorageManager::remove(handle.id)?;
    }
    Ok(())
}

pub fn resize_op(handle: &mut RustHandle, new_size: usize) -> Result<(), StorageError> {
    StorageManager::with(handle.id, |s| s.resize(new_size))??;
    handle.size_bytes = new_size;
    Ok(())
}

pub fn get_data_op(handle: &RustHandle) -> Result<*mut u8, StorageError> {
    StorageManager::with(handle.id, |s| s.data_ptr())
}

pub fn is_valid_op(handle: &RustHandle) -> bool {
    handle.is_valid() && StorageManager::contains(handle.id)
}
