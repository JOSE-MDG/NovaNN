//! Atomic unique ID generation for storage handles.
//!
//! Provides monotonically increasing 64-bit identifiers used to
//! look up storage entries in the global registry. ID `0` is reserved
//! as the sentinel invalid value, so the counter starts at `1`.

use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonically increasing ID counter.
///
/// Starts at `1` because `0` is reserved as the sentinel invalid ID.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Allocates the next unique identifier.
///
/// Uses relaxed ordering since the only requirement is uniqueness;
/// there is no need for happens-before relationships with other
/// atomic operations.
///
/// # Returns
///
/// A unique `u64` that has never been returned before (until wrap-around,
/// which is not handled — the system is assumed to not exhaust 2⁶⁴ IDs).
pub fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}
