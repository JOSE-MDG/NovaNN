//! Global registry for managing storage instances.

use crate::error::StorageError;
use crate::storage::RustStorage;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

static MANAGER: OnceLock<Mutex<HashMap<u64, RustStorage>>> = OnceLock::new();

/// Returns a reference to the global registry.
fn registry() -> &'static Mutex<HashMap<u64, RustStorage>> {
    MANAGER.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Manages storage instances globally.
pub struct StorageManager;

impl StorageManager {
    /// Inserts new storage into the registry.
    pub fn insert(id: u64, storage: RustStorage) -> Result<(), StorageError> {
        registry()
            .lock()
            .map_err(|_| StorageError::ManagerPoisoned)?
            .insert(id, storage);
        Ok(())
    }

    /// Apply a closure to the storage behind `id`, returning whatever the closure returns.
    pub fn with<F, R>(id: u64, f: F) -> Result<R, StorageError>
    where
        F: FnOnce(&mut RustStorage) -> R,
    {
        registry()
            .lock()
            .map_err(|_| StorageError::ManagerPoisoned)?
            .get_mut(&id)
            .map(f)
            .ok_or(StorageError::InvalidHandle)
    }

    /// Remove and return the storage, triggering Drop when the value is discarded.
    pub fn remove(id: u64) -> Result<RustStorage, StorageError> {
        registry()
            .lock()
            .map_err(|_| StorageError::ManagerPoisoned)?
            .remove(&id)
            .ok_or(StorageError::InvalidHandle)
    }

    /// Returns `true` if the registry contains the given id.
    pub fn contains(id: u64) -> bool {
        registry()
            .lock()
            .map(|map| map.contains_key(&id))
            .unwrap_or(false)
    }
}
