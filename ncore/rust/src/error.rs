/// Errors that can occur in the storage system.
#[derive(Debug, Clone, PartialEq)]
pub enum StorageError {
    /// Failed to allocate memory.
    AllocationFailed,
    /// Handle does not exist or has been invalidated.
    InvalidHandle,
    /// Alignment must be a power of two and non-zero.
    InvalidAlignment,
    /// Size must be greater than zero.
    InvalidSize,
    /// Failed to reallocate memory.
    ResizeFailed,
    /// Storage manager mutex was poisoned.
    ManagerPoisoned,
    /// Received null pointer.
    NullPointer,
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocationFailed => write!(f, "Memory allocation failed"),
            Self::InvalidHandle => write!(f, "Invalid or expired handle"),
            Self::InvalidAlignment => write!(f, "Alignment must be a power of two and non-zero"),
            Self::InvalidSize => write!(f, "Size must be greater than zero"),
            Self::ResizeFailed => write!(f, "Memory reallocation failed"),
            Self::ManagerPoisoned => write!(f, "Storage manager mutex was poisoned"),
            Self::NullPointer => write!(f, "Received null pointer"),
        }
    }
}

impl std::error::Error for StorageError {}
