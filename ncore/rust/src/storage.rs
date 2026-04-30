use crate::error::StorageError;
use std::alloc::{alloc, dealloc, realloc, Layout};

pub struct RustStorage {
    ptr: *mut u8,
    layout: Layout,
    pub size_bytes: usize,
    pub ref_count: usize,
}

// SAFETY: exclusive access is always enforced by the Mutex in StorageManager.
unsafe impl Send for RustStorage {}
unsafe impl Sync for RustStorage {}

impl RustStorage {
    pub fn allocate(size: usize, align: usize) -> Result<Self, StorageError> {
        if size == 0 {
            return Err(StorageError::InvalidSize);
        }

        let layout =
            Layout::from_size_align(size, align).map_err(|_| StorageError::InvalidAlignment)?;

        // SAFETY: layout is non-zero and valid.
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return Err(StorageError::AllocationFailed);
        }

        Ok(Self {
            ptr,
            layout,
            size_bytes: size,
            ref_count: 1,
        })
    }

    pub fn resize(&mut self, new_size: usize) -> Result<(), StorageError> {
        if new_size == 0 {
            return Err(StorageError::InvalidSize);
        }

        let new_layout = Layout::from_size_align(new_size, self.layout.align())
            .map_err(|_| StorageError::InvalidAlignment)?;

        // SAFETY: ptr was allocated with self.layout and is still valid.
        let new_ptr = unsafe { realloc(self.ptr, self.layout, new_size) };

        if new_ptr.is_null() {
            return Err(StorageError::ResizeFailed);
        }

        self.ptr = new_ptr;
        self.layout = new_layout;
        self.size_bytes = new_size;
        Ok(())
    }

    pub fn increment_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Returns `true` when ref_count hits zero — caller must free the storage.
    pub fn decrement_ref(&mut self) -> bool {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count == 0
    }

    pub fn data_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn align(&self) -> usize {
        self.layout.align()
    }
}

impl Drop for RustStorage {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was allocated with self.layout.
            unsafe { dealloc(self.ptr, self.layout) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
