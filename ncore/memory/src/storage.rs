//! Low-level memory allocation and management.
//!
//! Supports both CPU (system allocator) and GPU (C++ FFI) memory backends.
//! GPU allocations are tracked through an internal `Allocation` enum so that
//! explicit deallocation and [`Drop`] dispatch to the correct backend path.
//!
//! A [`RustStorage`] begins with one reference. The registry owns the storage
//! object while its reference count is non-zero. Normal lifecycle code calls
//! [`RustStorage::deallocate`] explicitly on the final release so backend
//! failures can be returned to the caller; [`Drop`] is retained as a final
//! cleanup path for unwinding and registry teardown.

use crate::counter::AtomicRefCounter;
use crate::error::StorageError;
use crate::ffi::cpp::{
    DeviceBuffer, DeviceKind, NovaError, deviceRelease, deviceReserve, deviceResize,
    getDeviceBackend,
};
use std::alloc::{Layout, alloc, dealloc, realloc};
use std::ffi::CStr;

/// Describes the allocator responsible for a storage block.
///
/// The variant is immutable for the lifetime of a storage object. It allows
/// resize and deallocation to dispatch to the same allocator that performed
/// the original allocation.
enum Allocation {
    /// Memory allocated through the platform allocator in `std::alloc`.
    Cpu {
        /// Memory layout used for allocation / deallocation.
        layout: Layout,
    },
    /// Memory allocated on a GPU device or as pinned host memory through the
    /// C++ FFI bridge.
    Gpu {
        /// Backend-specific buffer descriptor that must be passed to
        /// [`deviceRelease`] when the storage is freed.
        device_buf: DeviceBuffer,
    },
}

/// Low-level storage holding allocated memory and metadata.
///
/// Instances are created through [`RustStorage::allocate`] (CPU host memory)
/// or [`RustStorage::allocate_device`] (GPU device or pinned host memory) and
/// are explicitly deallocated on final release and also cleaned up when
/// dropped as a fallback.
///
/// # Thread safety
///
/// `RustStorage` is only accessed through the mutex in [`crate::manager`].
/// The `Send` and `Sync` implementations therefore rely on exclusive access
/// being enforced by the registry.
pub struct RustStorage {
    /// Pointer to the allocated memory (host or device).
    ptr: *mut u8,
    /// Allocation-backend details.
    alloc: Allocation,
    /// Size of the allocated memory in bytes.
    pub size_bytes: usize,
    /// Atomically tracked number of live owners represented by handles in the
    /// native core. Use [`AtomicRefCounter::get`] to read the value.
    pub ref_count: AtomicRefCounter,
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
    /// # Ownership
    ///
    /// The returned storage starts with a reference count of one. The caller
    /// is responsible for eventually calling [`Self::deallocate`] or allowing
    /// the value to be dropped.
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
            ref_count: AtomicRefCounter::new(1),
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
    /// * `pin_memory` - If `true`, allocate page-locked host memory.
    ///
    /// # Ownership
    ///
    /// The returned storage starts with a reference count of one. Device
    /// buffers remain owned by the C++ backend until [`Self::deallocate`]
    /// succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidSize`] when `size` is zero, or
    /// [`StorageError::DeviceError`] with the C++ status message when the
    /// backend operation fails.
    pub fn allocate_device(size: usize, pin_memory: bool) -> Result<Self, StorageError> {
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
        let status = unsafe { deviceReserve(size, &mut device_buf, pin_memory, kind) };

        if status.err != NovaError::Success {
            let msg = if status.message.is_null() {
                "Device operation failed with no error message".into()
            } else {
                // SAFETY: message points to a valid C string owned by the bridge.
                unsafe { CStr::from_ptr(status.message) }
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(StorageError::DeviceError {
                code: status.err,
                message: format!("Device allocation failed: {msg}"),
            });
        }

        // Get bytes before device_buf move
        let bytes = device_buf.bytes;

        Ok(Self {
            ptr: device_buf.ptr as *mut u8,
            alloc: Allocation::Gpu { device_buf },
            size_bytes: bytes,
            ref_count: AtomicRefCounter::new(1),
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
            Allocation::Gpu { device_buf } => {
                // SAFETY: device_buf was returned by a previous
                // deviceReserve call and has not been freed yet.
                let status = unsafe { deviceResize(device_buf as *mut DeviceBuffer, new_size) };

                if status.err != NovaError::Success {
                    let msg = if status.message.is_null() {
                        "Device operation failed with no error message".into()
                    } else {
                        // SAFETY: message points to a valid C string owned by the bridge.
                        unsafe { CStr::from_ptr(status.message) }
                            .to_string_lossy()
                            .into_owned()
                    };
                    return Err(StorageError::DeviceError {
                        code: status.err,
                        message: format!("Device resize failed: {msg}"),
                    });
                }

                self.ptr = device_buf.ptr as *mut u8;
                self.size_bytes = device_buf.bytes;
                Ok(())
            }
        }
    }

    /// Increments the reference count by one.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ReferenceCountOverflow`] if the count is
    /// already at `usize::MAX`.
    pub fn increment_ref(&self) -> Result<(), StorageError> {
        self.ref_count
            .try_increase()
            .map(|_| ())
            .ok_or(StorageError::ReferenceCountOverflow)
    }

    /// Decrements the reference count by one.
    ///
    /// Returns `true` when the count reaches zero. The caller must then call
    /// [`Self::deallocate`] before removing the storage from the registry.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidHandle`] if the count is already zero.
    pub fn decrement_ref(&self) -> Result<bool, StorageError> {
        let previous = self
            .ref_count
            .try_decrease()
            .ok_or(StorageError::InvalidHandle)?;
        Ok(previous == 1)
    }

    /// Releases the underlying allocation and reports backend failures.
    ///
    /// This operation is idempotent after a successful deallocation: a null
    /// data pointer returns `Ok(())`. On a device release failure, the pointer
    /// and backend descriptor remain available for the caller's recovery or
    /// retry policy.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::DeviceError`] when the C++ backend cannot free
    /// a device or pinned-host allocation.
    pub fn deallocate(&mut self) -> Result<(), StorageError> {
        if self.ptr.is_null() {
            return Ok(());
        }

        match &mut self.alloc {
            Allocation::Cpu { layout } => {
                // SAFETY: ptr was allocated with `layout` via std::alloc.
                unsafe { dealloc(self.ptr, *layout) };
                self.ptr = std::ptr::null_mut();
                Ok(())
            }
            Allocation::Gpu { device_buf } => {
                // SAFETY: device_buf was returned by a previous
                // deviceReserve call and has not been freed yet.
                let status = unsafe { deviceRelease(device_buf as *mut DeviceBuffer) };
                if status.err != NovaError::Success {
                    let msg = if status.message.is_null() {
                        "The device backend did not provide a failure description".to_string()
                    } else {
                        // SAFETY: the C++ bridge returns a valid NUL-terminated message.
                        unsafe { CStr::from_ptr(status.message) }
                            .to_string_lossy()
                            .into_owned()
                    };
                    return Err(StorageError::DeviceError {
                        code: status.err,
                        message: format!("Device release failed: {msg}"),
                    });
                }
                self.ptr = std::ptr::null_mut();
                Ok(())
            }
        }
    }

    /// Returns the raw pointer to the allocated data.
    ///
    /// The pointer is valid only while the storage remains allocated and must
    /// not be used after [`Self::deallocate`] succeeds.
    pub fn data_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Returns the alignment associated with the allocation.
    pub fn align(&self) -> usize {
        match &self.alloc {
            Allocation::Cpu { layout } => layout.align(),
            Allocation::Gpu { device_buf } => {
                if device_buf.is_pinned {
                    64
                } else {
                    512
                }
            }
        }
    }

    /// Returns `true` if the storage uses the C++ device-memory backend.
    ///
    /// This is also `true` for pinned host allocations because they are
    /// allocated and released by the active GPU backend.
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
        if let Err(error) = self.deallocate() {
            eprintln!("RustStorage cleanup failed: {error}");
        }
    }
}
