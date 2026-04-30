use std::sync::atomic::{AtomicU64, Ordering};

/// ID 0 is reserved as the sentinel "invalid" value.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Shared across FFI boundaries. Must stay `repr(C)` and trivially copyable.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustHandle {
    /// Unique key used to look up storage inside the manager.
    pub id: u64,
    /// Informational cache — authoritative value lives in RustStorage.
    pub size_bytes: usize,
    pub align: usize,
}

impl RustHandle {
    pub fn new(id: u64, size_bytes: usize, align: usize) -> Self {
        Self {
            id,
            size_bytes,
            align,
        }
    }

    pub fn invalid() -> Self {
        Self {
            id: 0,
            size_bytes: 0,
            align: 0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.id != 0
    }
}
