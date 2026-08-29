//! C-compatible type and function declarations for the C++ device-memory FFI
//! layer.
//!
//! Mirrors the types and `extern "C"` functions declared in
//! `csrc/ffi.hpp` so that Rust code can call into the C++ GPU
//! backend dispatch layer.
//!
//! These declarations are ABI contracts, not safe Rust abstractions. The
//! functions return [`NovaStatus`] instead of throwing, and the caller must
//! satisfy the pointer and ownership requirements documented on each
//! declaration.

use crate::status::NovaStatus;
use std::ffi::c_void;

/// GPU compute backend identifier.
///
/// Values match `DeviceKind_t` in `admin.hpp`.
#[repr(i8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// NVIDIA CUDA backend.
    CUDA = 0,
    /// AMD ROCm HIP backend.
    HIP = 1,
    /// No supported GPU backend is available.
    Null = 2,
}

/// Device-agnostic memory copy direction.
///
/// Values match `DeviceMemcpyKind` in `ffi.hpp`.
#[repr(i8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemcpyKind {
    /// Copy from host memory to device memory.
    HostToDevice = 1,
    /// Copy from device memory to host memory.
    DeviceToHost = 2,
    /// Copy between device allocations.
    DeviceToDevice = 3,
}

/// Backend-specific allocated buffer descriptor.
///
/// Layout matches `deviceBuffer_t` in `ffi.hpp`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceBuffer {
    /// Data pointer returned by the active backend.
    pub ptr: *mut c_void,
    /// Usable allocation size in bytes.
    pub bytes: usize,
    /// Whether the allocation is page-locked host memory.
    pub is_pinned: bool,
    /// Backend responsible for the allocation.
    pub device_kind: DeviceKind,
    /// Opaque backend descriptor owned by the C++ bridge.
    pub device_buf_ptr: *mut c_void,
}

unsafe extern "C" {
    /// Allocates a device or pinned-host buffer through the active backend.
    ///
    /// On success, `out_buf` receives a backend descriptor owned by the
    /// caller and it must eventually be passed to [`deviceRelease`]. On
    /// failure, the returned status contains the backend error and the output
    /// descriptor is reset by the C++ bridge.
    ///
    /// # Safety
    ///
    /// `out_buf` must be non-null and point to a valid, writable
    /// [`DeviceBuffer`]. `bytes` must be non-zero. The caller is responsible
    /// for eventually freeing a successful allocation with [`deviceRelease`].
    pub unsafe fn deviceReserve(
        bytes: usize,
        out_buf: *mut DeviceBuffer,
        pinned: bool,
        kind: DeviceKind,
    ) -> NovaStatus;

    /// Frees a buffer previously allocated with [`deviceReserve`].
    ///
    /// # Safety
    ///
    /// `buf` must be non-null, point to a buffer previously returned by
    /// [`deviceReserve`], and must not have been freed already. The descriptor
    /// is zeroed only when the release succeeds.
    pub unsafe fn deviceRelease(buf: *mut DeviceBuffer) -> NovaStatus;

    /// Resizes a device or pinned-host buffer while preserving content.
    ///
    /// Allocates a new buffer of `new_bytes`, copies the minimum of the
    /// old and new sizes, frees the old buffer, and updates
    /// [`DeviceBuffer::ptr`] and [`DeviceBuffer::bytes`] in-place on
    /// success.
    ///
    /// # Safety
    ///
    /// `buf` must be non-null, point to a buffer previously returned by
    /// [`deviceReserve`], and must not have been freed already. `new_bytes`
    /// must be non-zero.
    pub unsafe fn deviceResize(buf: *mut DeviceBuffer, new_bytes: usize) -> NovaStatus;

    /// Queries the active GPU compute backend.
    ///
    /// Returns [`DeviceKind::CUDA`], [`DeviceKind::HIP`], or
    /// [`DeviceKind::Null`] according to runtime detection.
    ///
    /// The result is selected by runtime probing and is used by the device
    /// allocation and transfer dispatch paths.
    pub unsafe fn getDeviceBackend() -> DeviceKind;
}
