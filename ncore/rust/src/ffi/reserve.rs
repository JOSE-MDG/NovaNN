//! C-compatible storage reservation.
//!
//! Exports [`reserve`] so that C code can allocate storage and
//! obtain a handle through a simple function call.

use crate::handle::RustHandle;
use crate::ops::reserve::reserve_op;

/// Allocates new storage with the given size and alignment.
///
/// Returns an invalid handle (ID = 0) on failure so that callers
/// can check success with [`is_valid_handle`].
///
/// # Safety
///
/// The returned [`RustHandle`] must be passed to [`release`] when
/// the caller is done with the storage to avoid memory leaks.
#[unsafe(no_mangle)]
pub extern "C" fn reserve(size: usize, align: usize) -> RustHandle {
    reserve_op(size, align).unwrap_or_else(|_| RustHandle::invalid())
}
