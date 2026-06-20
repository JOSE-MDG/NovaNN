//! Low-level memory allocation and management.
//!
//! Supports both CPU (system allocator) and GPU (C++ FFI) memory backends.
//! GPU allocations are tracked through an internal [`Allocation`] enum so that
//! [`Drop`] dispatches to the correct deallocation path.

use crate::error::StorageError;
use crate::ffi::cpp::{
    DeviceBuffer, DeviceKind, deviceRelease, deviceReserve, deviceResize, getDeviceBackend,
};
use std::alloc::{Layout, alloc, dealloc, realloc};
use std::ffi::CStr;

/// Describes how a storage block was allocated.
enum Allocation {
    /// Memory allocated through the system allocator (`std::alloc`).
    Cpu {
        /// Memory layout used for allocation / deallocation.
        layout: Layout,
    },
    /// Memory allocated on a GPU device (or pinned host) through the C++ FFI.
    Gpu {
        /// Backend-specific buffer descriptor that must be passed to
        /// [`deviceRelease`] when the storage is freed.
        device_buf: DeviceBuffer,
        /// Alignment requested at allocation time.
        alignment: usize,
    },
}

/// Low-level storage holding allocated memory and metadata.
///
/// Instances are created through [`RustStorage::allocate`] (CPU host memory)
/// or [`RustStorage::allocate_device`] (GPU device or pinned host memory) and
/// are automatically freed when dropped.
pub struct RustStorage {
    /// Pointer to the allocated memory (host or device).
    ptr: *mut u8,
    /// Allocation-backend details.
    alloc: Allocation,
    /// Size of the allocated memory in bytes.
    pub size_bytes: usize,
    /// Reference count for memory management.
    pub ref_count: usize,
}

// SAFETY: exclusive access is always enforced by the Mutex in StorageManager.
unsafe impl Send for RustStorage {}
unsafe impl Sync for RustStorage {}

impl RustStorage {
    /// Allocates new CPU storage via the system allocator.
    ///
    /// # Arguments
    ///
    /// * `size`  - Number of bytes to allocate. Must be non-zero.
    /// * `align` - Required memory alignment. Must be a power of two.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidSize`] when `size` is zero,
    /// [`StorageError::InvalidAlignment`] when `align` is not a power of two,
    /// or [`StorageError::AllocationFailed`] when the system allocator returns
    /// a null pointer.
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
            alloc: Allocation::Cpu { layout },
            size_bytes: size,
            ref_count: 1,
        })
    }

    /// Allocates GPU device memory (or pinned host memory) through the
    /// C++ FFI layer.
    ///
    /// When `pin_memory` is `true` the memory is allocated as page-locked
    /// host memory via the active GPU backend; otherwise it is allocated
    /// on the GPU device itself.
    ///
    /// # Arguments
    ///
    /// * `size`       - Number of bytes to allocate. Must be non-zero.
    /// * `alignment`  - Required alignment (passed to the C++ backend).
    /// * `pin_memory` - If `true`, allocate page-locked host memory.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidSize`] when `size` is zero, or
    /// [`StorageError::DeviceError`] with the C++ status message when the
    /// backend operation fails.
    pub fn allocate_device(
        size: usize,
        alignment: usize,
        pin_memory: bool,
    ) -> Result<Self, StorageError> {
        if size == 0 {
            return Err(StorageError::InvalidSize);
        }

        let mut device_buf = DeviceBuffer {
            ptr: std::ptr::null_mut(),
            bytes: 0,
            is_pinned: false,
            device_kind: DeviceKind::Null,
            device_buf_ptr: std::ptr::null_mut(),
        };

        let kind = unsafe { getDeviceBackend() };

        // SAFETY: device_buf is a valid, writable stack allocation.
        let status = unsafe { deviceReserve(size, &mut device_buf, pin_memory, alignment, kind) };

        if status.code != 0 {
            let msg = if status.message.is_null() {
                "Unknown device error message, the message null".into()
            } else {
                // SAFETY: message points to a static C string literal.
                unsafe { CStr::from_ptr(status.message) }
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(StorageError::DeviceError(msg));
        }

        // Get bytes before device_buf move
        let bytes = device_buf.bytes;

        Ok(Self {
            ptr: device_buf.ptr as *mut u8,
            alloc: Allocation::Gpu {
                device_buf,
                alignment,
            },
            size_bytes: bytes,
            ref_count: 1,
        })
    }

    /// Resizes the allocated memory to the new size, preserving existing data.
    ///
    /// For CPU storage the underlying [`realloc`] is used; for GPU storage the
    /// operation is forwarded to [`deviceResize`] which allocates a new
    /// buffer, copies the minimum of the old and new sizes, and frees the old
    /// buffer atomically on the device stream.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidSize`] when `new_size` is zero,
    /// [`StorageError::InvalidAlignment`] when the new layout is misaligned
    /// (CPU only), [`StorageError::ResizeFailed`] when the system realloc
    /// returns null, or [`StorageError::DeviceError`] with the backend error
    /// message when the GPU realloc fails.
    pub fn resize(&mut self, new_size: usize) -> Result<(), StorageError> {
        if new_size == 0 {
            return Err(StorageError::InvalidSize);
        }

        match &mut self.alloc {
            Allocation::Cpu { layout } => {
                let new_layout = Layout::from_size_align(new_size, layout.align())
                    .map_err(|_| StorageError::InvalidAlignment)?;

                // SAFETY: ptr was allocated with `layout` and is still valid.
                let new_ptr = unsafe { realloc(self.ptr, *layout, new_size) };

                if new_ptr.is_null() {
                    return Err(StorageError::ResizeFailed);
                }

                self.ptr = new_ptr;
                *layout = new_layout;
                self.size_bytes = new_size;
                Ok(())
            }
            Allocation::Gpu {
                device_buf,
                alignment,
            } => {
                // SAFETY: device_buf was returned by a previous
                // deviceReserve call and has not been freed yet.
                let status =
                    unsafe { deviceResize(device_buf as *mut DeviceBuffer, new_size, *alignment) };

                if status.code != 0 {
                    let msg = if status.message.is_null() {
                        "Unknown device error message, the message null".into()
                    } else {
                        // SAFETY: message points to a static C string literal.
                        unsafe { CStr::from_ptr(status.message) }
                            .to_string_lossy()
                            .into_owned()
                    };
                    return Err(StorageError::DeviceError(msg));
                }

                self.ptr = device_buf.ptr as *mut u8;
                self.size_bytes = device_buf.bytes;
                Ok(())
            }
        }
    }

    /// Increments the reference count.
    pub fn increment_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrements the reference count.
    ///
    /// Returns `true` when ref_count hits zero — caller must free the storage.
    pub fn decrement_ref(&mut self) -> bool {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count == 0
    }

    /// Returns a raw pointer to the allocated data.
    pub fn data_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Returns the memory alignment.
    pub fn align(&self) -> usize {
        match &self.alloc {
            Allocation::Cpu { layout } => layout.align(),
            Allocation::Gpu { alignment, .. } => *alignment,
        }
    }

    /// Returns `true` if the storage was allocated on a GPU device.
    pub fn is_device_memory(&self) -> bool {
        matches!(self.alloc, Allocation::Gpu { .. })
    }

    /// Returns `true` if the storage resides in pinned (page-locked) host
    /// memory.
    pub fn is_pinned(&self) -> bool {
        match &self.alloc {
            Allocation::Cpu { .. } => false,
            Allocation::Gpu { device_buf, .. } => device_buf.is_pinned,
        }
    }
}

impl Drop for RustStorage {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        match &mut self.alloc {
            Allocation::Cpu { layout } => {
                // SAFETY: ptr was allocated with `layout` via std::alloc.
                unsafe { dealloc(self.ptr, *layout) };
            }
            Allocation::Gpu { device_buf, .. } => {
                // SAFETY: device_buf was returned by a previous
                // deviceReserve call and has not been freed yet.
                let status = unsafe { deviceRelease(device_buf as *mut DeviceBuffer) };
                if status.code != 0 {
                    let msg = if status.message.is_null() {
                        "Unknown device error message, the message null".to_string()
                    } else {
                        // SAFETY: message points to a static C string literal.
                        unsafe { CStr::from_ptr(status.message) }
                            .to_string_lossy()
                            .into_owned()
                    };
                    eprintln!(
                        "RustStorage::drop: deviceRelease failed (code={}): {msg}",
                        status.code
                    );
                }
            }
        }

        self.ptr = std::ptr::null_mut();
    }
}
