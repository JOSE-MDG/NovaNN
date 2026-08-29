//! Error types for fallible storage operations.
//!
//! [`StorageError`] is used inside the Rust implementation. The FFI layer
//! converts it to [`crate::NovaStatus`] immediately before crossing into C or
//! C++, preserving both a machine-readable code and a readable diagnostic.

use crate::status::NovaError;

/// Errors that can occur in the storage system.
#[derive(Debug, Clone, PartialEq)]
pub enum StorageError {
    /// The selected allocator could not provide the requested memory.
    AllocationFailed,
    /// The handle does not exist or has already been invalidated.
    InvalidHandle,
    /// The requested alignment is zero or not a power of two.
    InvalidAlignment,
    /// The requested memory layout is invalid.
    InvalidMemoryLayout,
    /// The requested allocation or resize size is zero.
    InvalidSize,
    /// The CPU allocator could not resize the allocation.
    ResizeFailed,
    /// The global storage registry mutex is poisoned and cannot be used.
    ManagerPoisoned,
    /// A required pointer argument was null.
    NullPointer,
    /// The device string is not `"cpu"` or `"device"`.
    InvalidDevice,
    /// Pinned host memory was requested for a GPU device allocation.
    PinnedMemoryOnDevice,
    /// The storage reference count would overflow on increment.
    ReferenceCountOverflow,
    /// The C++ device-memory FFI layer returned a failure status.
    DeviceError {
        /// Error code returned by the C++ device backend.
        code: NovaError,
        /// Human-readable diagnostic returned by the backend.
        message: String,
    },
}

impl StorageError {
    /// Maps an internal storage error to the public NovaNN error code.
    pub(crate) fn code(&self) -> NovaError {
        match self {
            Self::AllocationFailed => NovaError::OutOfMemory,
            Self::InvalidHandle => NovaError::InvalidHandle,
            Self::InvalidAlignment => NovaError::InvalidAlignment,
            Self::InvalidMemoryLayout => NovaError::InvalidMemoryLayout,
            Self::InvalidSize => NovaError::InvalidValue,
            Self::ResizeFailed => NovaError::OutOfMemory,
            Self::ManagerPoisoned | Self::ReferenceCountOverflow => NovaError::InternalError,
            Self::NullPointer => NovaError::InvalidPointer,
            Self::InvalidDevice => NovaError::InvalidDevice,
            Self::PinnedMemoryOnDevice => NovaError::InvalidValue,
            Self::DeviceError { code, .. } if *code != NovaError::Success => *code,
            Self::DeviceError { .. } => NovaError::InternalError,
        }
    }
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocationFailed => write!(f, "Memory allocation failed"),
            Self::InvalidHandle => write!(f, "Invalid or expired handle"),
            Self::InvalidAlignment => write!(f, "Alignment must be a power of two and non-zero"),
            Self::InvalidMemoryLayout => write!(f, "The requested memory layout is invalid"),
            Self::InvalidSize => write!(f, "Size must be greater than zero"),
            Self::ResizeFailed => write!(f, "Memory reallocation failed"),
            Self::ManagerPoisoned => write!(f, "Storage manager is unavailable"),
            Self::NullPointer => write!(f, "Storage handle pointer must not be null"),
            Self::InvalidDevice => write!(f, "Device must be \"cpu\" or \"device\""),
            Self::PinnedMemoryOnDevice => {
                write!(
                    f,
                    "Pinned host memory cannot be requested for a device allocation"
                )
            }
            Self::ReferenceCountOverflow => write!(f, "Storage reference count overflowed"),
            Self::DeviceError { message, .. } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for StorageError {}
