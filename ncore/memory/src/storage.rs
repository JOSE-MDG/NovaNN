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
//!
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
    /// GPU ordinal captured at reservation time, or `None` for CPU memory.
    device_id: Option<i32>,
    /// Theoretical size asked by the caller, in bytes.
    pub requested_bytes: usize,
    /// Real reservation in bytes. Greater than or equal to `requested_bytes`.
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
    /// * `size`      - Reservation size in bytes. Must be non-zero.
    /// * `requested` - Theoretical size asked by the caller. Must be non-zero
    ///   and not greater than `size`.
    /// * `align`     - Required memory alignment. Must be a power of two.
    ///
    /// # Ownership
    ///
    /// The returned storage starts with a reference count of one. The caller
    /// is responsible for eventually calling [`Self::deallocate`] or allowing
    /// the value to be dropped.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidSize`] when `size` or `requested` is
    /// zero or when `requested` exceeds `size`,
    /// [`StorageError::InvalidAlignment`] when `align` is zero or not a
    /// power of two,
    /// [`StorageError::InvalidMemoryLayout`] when `align` or `size` cannot be
    /// represented by the platform allocator, or
    /// [`StorageError::AllocationFailed`] when the system allocator returns
    /// a null pointer.
    pub fn allocate(size: usize, requested: usize, align: usize) -> Result<Self, StorageError> {
        if size == 0 || requested == 0 || requested > size {
            return Err(StorageError::InvalidSize);
        }

        if ((align > 0) && (align & (align - 1)) != 0) || (align == 0) {
            return Err(StorageError::InvalidAlignment);
        }

        let layout =
            Layout::from_size_align(size, align).map_err(|_| StorageError::InvalidMemoryLayout)?;

        // SAFETY: layout is non-zero and valid.
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return Err(StorageError::AllocationFailed);
        }

        Ok(Self {
            ptr,
            alloc: Allocation::Cpu { layout },
            device_id: None,
            requested_bytes: requested,
            size_bytes: size,
            ref_count: AtomicRefCounter::new(1),
        })
    }

    /// Allocates GPU device memory (or pinned host memory) through the
    /// C++ FFI layer.
    ///
    /// When `pin_memory` is `true` the memory is allocated as page-locked
    /// host memory via the active GPU backend; otherwise it is allocated
    /// on the GPU device itself. Alignment is determined by the backend.
    ///
    /// # Arguments
    ///
    /// * `size`       - Reservation size in bytes, already rounded to its
    ///   size class by the caller. Must be non-zero.
    /// * `requested`  - Theoretical size asked by the caller. Must be
    ///   non-zero and not greater than `size`.
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
    /// Returns [`StorageError::InvalidSize`] when `size` or `requested` is
    /// zero or when `requested` exceeds `size`, or [`StorageError::DeviceError`]
    /// with the C++ status message when the backend operation fails.
    pub fn allocate_device(
        size: usize,
        requested: usize,
        pin_memory: bool,
    ) -> Result<Self, StorageError> {
        if size == 0 || requested == 0 || requested > size {
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
        let mut device_id = -1;
        let status: crate::NovaStatus = unsafe {
            deviceReserve(
                size,
                &mut device_buf,
                pin_memory,
                kind,
                &mut device_id,
                std::ptr::null_mut(),
            )
        };

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

        if bytes == 0 || bytes < requested {
            // The backend reported success with an unusable size: release
            // the buffer instead of registering a reservation that violates
            // the requested <= bytes contract.
            unsafe {
                deviceRelease(
                    &mut device_buf as *mut DeviceBuffer,
                    device_id,
                    std::ptr::null_mut(),
                )
            };
            return Err(StorageError::InvalidSize);
        }

        Ok(Self {
            ptr: device_buf.ptr as *mut u8,
            alloc: Allocation::Gpu { device_buf },
            device_id: Some(device_id),
            requested_bytes: requested,
            size_bytes: bytes,
            ref_count: AtomicRefCounter::new(1),
        })
    }

    /// Resizes the allocated memory to the new theoretical size, preserving
    /// existing data.
    ///
    /// For CPU storage the underlying [`realloc`] is used with the exact
    /// size; for GPU storage the size is first rounded to its size class
    /// with the same rules as the initial reservation, then forwarded to
    /// [`deviceResize`] which allocates a new buffer, copies the minimum of
    /// the old and new sizes, and frees the old buffer through the same
    /// stream.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidSize`] when `requested` is zero, when
    /// it cannot be rounded, or when `requested` exceeds the current real
    /// size contract, [`StorageError::InvalidAlignment`] when the new layout
    /// is misaligned (CPU only), [`StorageError::ResizeFailed`] when the
    /// system realloc returns null, or [`StorageError::DeviceError`] with the
    /// backend error message when the GPU realloc fails.
    pub fn resize(&mut self, requested: usize) -> Result<(), StorageError> {
        if requested == 0 {
            return Err(StorageError::InvalidSize);
        }

        match &mut self.alloc {
            Allocation::Cpu { layout } => {
                let new_layout = Layout::from_size_align(requested, layout.align())
                    .map_err(|_| StorageError::InvalidAlignment)?;

                // SAFETY: ptr was allocated with `layout` and is still valid.
                let new_ptr = unsafe { realloc(self.ptr, *layout, requested) };

                if new_ptr.is_null() {
                    return Err(StorageError::ResizeFailed);
                }

                self.ptr = new_ptr;
                *layout = new_layout;
                self.requested_bytes = requested;
                self.size_bytes = requested;
                Ok(())
            }
            Allocation::Gpu { device_buf } => {
                let pinned = device_buf.is_pinned;
                let rounded = crate::rounding::resolve(requested, true, pinned)
                    .ok_or(StorageError::InvalidSize)?;
                // SAFETY: device_buf was returned by a previous
                // deviceReserve call and has not been freed yet.
                let device_id = self.device_id.unwrap_or(-1);
                let previous_ptr = self.ptr;
                // A null stream selects the device engine stream, which owns
                // the buffer: the backend rejects a resize submitted on a
                // different stream.
                let status: crate::NovaStatus = unsafe {
                    deviceResize(
                        device_buf as *mut DeviceBuffer,
                        rounded,
                        device_id,
                        std::ptr::null_mut(),
                    )
                };

                if device_buf.ptr != previous_ptr.cast() {
                    self.ptr = device_buf.ptr.cast();
                    self.size_bytes = device_buf.bytes;
                }

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

                if device_buf.bytes == 0 || device_buf.bytes < requested {
                    // The backend reported success with an unusable size.
                    // The previous buffer is already gone; keep the new
                    // pointer so the storage is not left dangling.
                    self.ptr = device_buf.ptr as *mut u8;
                    self.size_bytes = device_buf.bytes;
                    return Err(StorageError::InvalidSize);
                }

                self.ptr = device_buf.ptr as *mut u8;
                self.requested_bytes = requested;
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
    /// This operation is idempotent after deallocation: a null data pointer
    /// returns `Ok(())`. If the backend accepted an asynchronous free but a
    /// later synchronization reported an error, the descriptor is consumed
    /// and a retry resolves the retained registry entry without freeing the
    /// pointer twice. Errors before the backend accepts the free leave the
    /// allocation available for retry.
    ///
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
                let device_id = self.device_id.unwrap_or(-1);
                // A null stream selects the device engine stream, which owns
                // the buffer: the backend rejects a release submitted on a
                // different stream.
                let status = unsafe {
                    deviceRelease(
                        device_buf as *mut DeviceBuffer,
                        device_id,
                        std::ptr::null_mut(),
                    )
                };
                if device_buf.ptr.is_null() {
                    self.ptr = std::ptr::null_mut();
                }
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

    /// Largest power of two dividing the pointer address; null yields 1.
    fn proven_alignment(ptr: *const u8) -> usize {
        if ptr.is_null() {
            return 1;
        }
        let addr = ptr as usize;
        let low = addr & addr.wrapping_neg();
        if low == 0 { 1 } else { low }
    }

    /// Returns the alignment associated with the allocation.
    ///
    /// CPU memory reports the requested layout alignment; device and
    /// pinned host memory report the backend alignment.
    pub fn align(&self) -> usize {
        match &self.alloc {
            Allocation::Cpu { layout } => layout.align(),
            Allocation::Gpu { .. } => Self::proven_alignment(self.ptr as *const u8),
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
