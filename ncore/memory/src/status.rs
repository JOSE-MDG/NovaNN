//! Status values shared by the Rust storage layer and its C/C++ ABI.
//!
//! [`NovaStatus`] is the sole error-reporting object used by the exported
//! storage operations. Its layout and enum discriminants mirror the C types
//! declared in `ncore/core/status.h`; changes to either side must therefore
//! be made together.

use crate::error::StorageError;
use std::cell::RefCell;
use std::ffi::{CString, c_char};

/// NovaNN error codes returned across the C ABI.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NovaError {
    /// The operation completed successfully.
    Success = 0,
    /// An argument has an invalid value.
    InvalidValue = 1,
    /// A tensor argument is invalid.
    InvalidTensor = 2,
    /// A pointer argument is null or invalid.
    InvalidPointer = 3,
    /// A data type argument is invalid.
    InvalidDtype = 4,
    /// A device argument is invalid.
    InvalidDevice = 5,
    /// A tensor dimension count is invalid.
    InvalidNdims = 6,
    /// A requested memory alignment is invalid.
    InvalidAlignment = 7,
    /// A tensor shape is invalid.
    InvalidShape = 8,
    /// A tensor index is invalid.
    InvalidIndex = 9,
    /// A requested thread count is invalid.
    InvalidNumThreads = 10,
    /// An operation would exceed a buffer boundary.
    BufferOverflow = 11,
    /// The requested allocation could not be completed.
    OutOfMemory = 12,
    /// A storage reservation failed.
    ReserveError = 13,
    /// A storage release failed.
    ReleaseError = 14,
    /// A storage resize failed.
    ResizeError = 15,
    /// A device transfer failed.
    TransferError = 16,
    /// A host-to-device transfer failed.
    TransferH2DError = 17,
    /// A device-to-host transfer failed.
    TransferD2HError = 18,
    /// A transfer direction is invalid.
    InvalidTransfDirection = 19,
    /// No suitable device is available.
    DeviceNotAvailable = 20,
    /// The requested device has not been initialized.
    DeviceNotInitialized = 21,
    /// An external device API reported a failure.
    ExternalDeviceError = 22,
    /// The requested backend was not compiled into the library.
    BackendNotCompiled = 23,
    /// The requested backend does not support the operation.
    BackendNotSupported = 24,
    /// The current operating system cannot support the operation.
    OsPlatformNotSupported = 25,
    /// A data type is not supported by the selected backend.
    DtypeNotSupported = 26,
    /// A type-casting operation is not supported.
    CastNotSupported = 27,
    /// Tensor shapes are incompatible for the requested operation.
    ShapeMismatch = 28,
    /// A device or storage resource handle is invalid.
    InvalidResourceHandle = 29,
    /// A device kernel failed to launch.
    KernelLaunchError = 30,
    /// The requested operation has not been implemented.
    NotImplemented = 31,
    /// An internal invariant or subsystem failed.
    InternalError = 32,
    /// An unspecified runtime failure occurred.
    RuntimeError = 33,
}

/// C-compatible status containing an error code and diagnostic message.
///
/// A successful operation sets [`err`](Self::err) to [`NovaError::Success`].
/// On failure, [`message`](Self::message) points to a human-readable,
/// NUL-terminated diagnostic string. The message is intended for immediate
/// consumption by the caller and remains valid until the next storage
/// operation on the same thread.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NovaStatus {
    /// Operation result code. This field has the same discriminant values as
    /// the C `novaError_t` enumeration.
    pub err: NovaError,
    /// Pointer to a NUL-terminated diagnostic message, or a success message
    /// when [`err`](Self::err) is [`NovaError::Success`].
    pub message: *const c_char,
}

static SUCCESS_MESSAGE: &[u8] = b"Operation completed successfully\0";
static FALLBACK_MESSAGE: &[u8] = b"Unable to construct a storage error message\0";

thread_local! {
    static STATUS_MESSAGE: RefCell<Option<CString>> = const { RefCell::new(None) };
}

impl NovaStatus {
    /// Creates a success status using the static success message.
    pub(crate) const fn success() -> Self {
        Self {
            err: NovaError::Success,
            message: SUCCESS_MESSAGE.as_ptr().cast(),
        }
    }

    /// Converts a Rust storage error to its stable C ABI representation.
    ///
    /// The formatted message is stored in thread-local state so a returned
    /// [`NovaStatus`] remains a small, copyable C-compatible value. Any NUL
    /// byte in an error message is escaped before constructing the C string.
    pub(crate) fn from_error(error: &StorageError) -> Self {
        let message = error.to_string().replace('\0', "\\0");
        let code = error.code();
        let cmessage = CString::new(message).unwrap_or_default();

        let message_ptr = STATUS_MESSAGE.with(|slot| match slot.try_borrow_mut() {
            Ok(mut current) => {
                *current = Some(cmessage);
                current
                    .as_ref()
                    .map_or(FALLBACK_MESSAGE.as_ptr().cast(), |value| value.as_ptr())
                    .cast()
            }
            Err(_) => FALLBACK_MESSAGE.as_ptr().cast(),
        });

        Self {
            err: code,
            message: message_ptr,
        }
    }
}
