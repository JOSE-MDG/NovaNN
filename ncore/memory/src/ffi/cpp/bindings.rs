//! C-compatible type and function declarations for the C++ device-memory FFI
//! layer.
//!
//! Mirrors the types and `extern "C"` functions declared in
//! `csrc/ffi.hpp` so that Rust code can call into the C++ GPU
//! backend dispatch layer.

use std::ffi::c_char;
use std::ffi::c_void;

/// GPU compute backend identifier.
///
/// Values match `DeviceKind_t` in `admin.hpp`.
#[repr(i8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    CUDA = 0,
    HIP = 1,
    Null = 2,
}

/// Device-agnostic memory copy direction.
///
/// Values match `DeviceMemcpyKind` in `ffi.hpp`.
#[repr(i8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemcpyKind {
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

/// Backend-specific allocated buffer descriptor.
///
/// Layout matches `deviceBuffer_t` in `ffi.hpp`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceBuffer {
    pub ptr: *mut c_void,
    pub bytes: usize,
    pub is_pinned: bool,
    pub device_kind: DeviceKind,
    pub device_buf_ptr: *mut c_void,
}

/// Error codes returned by the NovaNN runtime.
///
/// Values match `novaError_t` in `core/status.h`.  The enumerators are
/// organised into logical groups (Parameters, Memory, Transfers, etc.)
/// to aid in categorisation and debugging.
#[repr(i8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NovaError {
    Success = 0,

    /* Invalid parameters */
    InvalidValue = 1,
    InvalidTensor = 2,
    InvalidPointer = 3,
    InvalidDtype = 4,
    InvalidDevice = 5,
    InvalidNdims = 6,
    InvalidAlignment = 7,
    InvalidShape = 8,
    InvalidIndex = 9,
    InvalidNumThreads = 10,

    /* Memory */
    BufferOverflow = 11,
    OutOfMemory = 12,
    ReserveError = 13,
    ReleaseError = 14,
    ResizeError = 15,

    /* Data transfer */
    TransferError = 16,
    TransferH2DError = 17,
    TransferD2HError = 18,
    InvalidTransfDirection = 19,

    /* Device / Backend */
    DeviceNotAvailable = 20,
    DeviceNotInitialized = 21,
    ExternalDeviceError = 22,
    BackendNotCompiled = 23,
    BackendNotSupported = 24,

    /* OS/Platform */
    OsPlatformNotSupported = 25,

    /* Dtype/Cast */
    DtypeNotSupported = 26,
    CastNotSupported = 27,
    ShapeMismatch = 28,

    /* GPU-specific */
    InvalidResourceHandle = 29,
    KernelLaunchError = 30,

    /* Internal */
    NotImplemented = 31,
    InternalError = 32,

    /* General */
    RuntimeError = 33,
}

/// Container for an error code and an optional detailed message.
///
/// Values match `novaStatus_t` in `core/status.h`.  Used in
/// high-level APIs where additional context about the failure is
/// required.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NovaStatus {
    pub err: NovaError,
    pub message: *const c_char,
}

unsafe extern "C" {
    /// Allocate a device or pinned-host buffer through the active backend.
    ///
    /// # Safety
    ///
    /// `out_buf` must be non-null and point to a valid, writable
    /// [`DeviceBuffer`]. The caller is responsible for eventually freeing
    /// the buffer with [`deviceRelease`].
    pub unsafe fn deviceReserve(
        bytes: usize,
        out_buf: *mut DeviceBuffer,
        pinned: bool,
        align: usize,
        kind: DeviceKind,
    ) -> NovaStatus;

    /// Free a buffer previously allocated with [`deviceReserve`].
    ///
    /// # Safety
    ///
    /// `buf` must be non-null, point to a buffer previously returned by
    /// [`deviceReserve`], and must not have been freed already.
    pub unsafe fn deviceRelease(buf: *mut DeviceBuffer) -> NovaStatus;

    /// Reallocate a device or pinned-host buffer, preserving content.
    ///
    /// Allocates a new buffer of `new_bytes` (rounded up to `align`),
    /// copies the minimum of the old and new sizes, frees the old buffer,
    /// and updates [`DeviceBuffer::ptr`] and [`DeviceBuffer::bytes`]
    /// in-place on success.
    ///
    /// # Safety
    ///
    /// `buf` must be non-null, point to a buffer previously returned by
    /// [`deviceReserve`], and must not have been freed already.
    pub unsafe fn deviceResize(
        buf: *mut DeviceBuffer,
        new_bytes: usize,
        align: usize,
    ) -> NovaStatus;

    /// Query the active GPU compute backend.
    ///
    /// Returns [`DeviceKind::CUDA`], [`DeviceKind::HIP`], or
    /// [`DeviceKind::Null`] according to runtime detection.
    ///
    /// Corresponds to `getDeviceBackend()` in `admin.hpp`.
    pub unsafe fn getDeviceBackend() -> DeviceKind;
}
