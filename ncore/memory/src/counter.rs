//! Atomic reference-count storage used by [`crate::storage::RustStorage`].
//!
//! The counter uses relaxed atomic operations because the storage registry
//! provides the synchronization required for allocation ownership. The
//! checked methods are used by the lifecycle layer so a failed increment or
//! decrement never wraps the counter.

use std::sync::atomic::{AtomicUsize, Ordering};

/// An atomically updated reference count.
///
/// The counter stores the number of live owners of a storage allocation. It
/// is initialized explicitly and does not impose a policy about when the
/// underlying allocation should be freed; that decision belongs to the
/// storage lifecycle layer.
pub struct AtomicRefCounter {
    count: AtomicUsize,
}

impl AtomicRefCounter {
    /// Creates a counter with the supplied initial value.
    pub const fn new(initial: usize) -> Self {
        Self {
            count: AtomicUsize::new(initial),
        }
    }

    /// Increments the counter unless it is already at `usize::MAX`.
    ///
    /// Returns the previous value on success, or `None` when incrementing
    /// would overflow.
    pub fn try_increase(&self) -> Option<usize> {
        self.count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .ok()
    }

    /// Decrements the counter unless it is already zero.
    ///
    /// Returns the previous value on success, or `None` when decrementing
    /// would underflow.
    pub fn try_decrease(&self) -> Option<usize> {
        self.count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_sub(1)
            })
            .ok()
    }

    /// Returns the current counter value.
    pub fn get(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}
