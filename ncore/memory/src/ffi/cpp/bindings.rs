//! C-compatible type and function declarations for the C++ device-memory FFI
//! layer.
//!
//! Mirrors the types and `extern "C"` functions declared in
//! `ncore/include/ncore/memory/ffi.hpp` so that Rust code can call into the C++ GPU
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
/// Values match `DeviceKind_t` in `ncore/include/ncore/memory/admin.hpp`.
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
/// Values match `DeviceMemcpyKind` in `ncore/include/ncore/memory/ffi.hpp`.
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
/// Layout matches `deviceBuffer_t` in `ncore/include/ncore/memory/ffi.hpp`.
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
    /// `device_id` receives the runtime device selected for the allocation;
    /// it must be saved and passed back to [`deviceRelease`] and
    /// [`deviceResize`] so the pool is addressed correctly.
    ///
    /// `stream` is the native-border stream handle. Null selects the device
    /// engine stream.
    ///
    /// # Safety
    ///
    /// `out_buf` must be non-null and point to a valid, writable
    /// [`DeviceBuffer`]. `bytes` must be non-zero. `device_id` must be
    /// non-null and point to writable storage. The caller is responsible
    /// for eventually freeing a successful allocation with [`deviceRelease`].
    pub unsafe fn deviceReserve(
        bytes: usize,
        out_buf: *mut DeviceBuffer,
        pinned: bool,
        kind: DeviceKind,
        device_id: *mut i32,
        stream: *mut c_void,
    ) -> NovaStatus;

    /// Frees a buffer previously allocated with [`deviceReserve`].
    ///
    /// `device_id` must match the ordinal returned by the reserve call that
    /// produced `buf`, because device memory is returned through that
    /// device's pool.
    ///
    /// `stream` is the native-border stream handle. Null selects the device
    /// engine stream.
    ///
    /// # Safety
    ///
    /// `buf` must be non-null, point to a buffer previously returned by
    /// [`deviceReserve`], and must not have been freed already. The descriptor
    /// is zeroed only when the release succeeds.
    pub unsafe fn deviceRelease(
        buf: *mut DeviceBuffer,
        device_id: i32,
        stream: *mut c_void,
    ) -> NovaStatus;

    /// Resizes a device or pinned-host buffer while preserving content.
    ///
    /// Allocates a new buffer of `new_bytes`, copies the minimum of the
    /// old and new sizes, frees the old buffer, and updates
    /// [`DeviceBuffer::ptr`] and [`DeviceBuffer::bytes`] in-place on
    /// success.
    ///
    /// `device_id` must match the ordinal returned by the reserve call that
    /// produced `buf`.
    ///
    /// # Safety
    ///
    /// `buf` must be non-null, point to a buffer previously returned by
    /// [`deviceReserve`], and must not have been freed already. `new_bytes`
    /// must be non-zero.
    pub unsafe fn deviceResize(
        buf: *mut DeviceBuffer,
        new_bytes: usize,
        device_id: i32,
        stream: *mut c_void,
    ) -> NovaStatus;

    /// Queries the active GPU compute backend.
    ///
    /// Returns [`DeviceKind::CUDA`], [`DeviceKind::HIP`], or
    /// [`DeviceKind::Null`] according to runtime detection.
    ///
    /// The result is selected by runtime probing and is used by the device
    /// allocation and transfer dispatch paths.
    pub unsafe fn getDeviceBackend() -> DeviceKind;

    /// Trims reclaimable backing from one backend device pool.
    ///
    /// The pool keeps at least `min_bytes_to_keep` resident bytes and
    /// returns the rest to the runtime. Memory that live allocations
    /// still hold is never reclaimed. When the device does not expose a
    /// native memory pool the call is a no-op and reports success.
    ///
    /// `kind` selects the backend and `device_id` the runtime ordinal
    /// owning the pool.
    ///
    /// # Safety
    ///
    /// `kind` must name a backend compiled into the library, and
    /// `device_id` must be a valid ordinal for that backend.
    pub unsafe fn deviceEmptyCache(
        kind: DeviceKind,
        device_id: i32,
        min_bytes_to_keep: usize,
    ) -> NovaStatus;
}
