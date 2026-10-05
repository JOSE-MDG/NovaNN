//! Size-class rounding for device-backed reservations.
//!
//! Mirrors the public rounding surface in `ncore/core/storage.h`: device
//! requests use a 512-byte step up to 10 MiB and a 2 MiB step above, while
//! pinned requests use a 4096-byte step up to 16 KiB and geometric classes
//! (1x, 1.25x, 1.5x, 1.75x of each power of two) above. Plain host
//! reservations are identity. The same constants live in
//! `ncore/src/core/storage.c`. Both sides share test vectors through the C
//! suite.
//!
//! The resize path intentionally omits the backend-active gate used on the
//! reservation path: it operates on live storage, so the backend backing it
//! is proven present and rounding must stay stable.

/// Device requests at or below this size use the 512-byte step.
pub(crate) const DEVICE_LARGE_THRESHOLD: usize = 10 << 20;
/// Pinned requests at or below this size use the 4096-byte step.
pub(crate) const PINNED_LINEAR_LIMIT: usize = 16 << 10;
/// Small-step alignment for device memory.
pub(crate) const DEVICE_SMALL: usize = 512;
/// Small-step alignment for pinned host memory.
pub(crate) const PINNED_SMALL: usize = 4096;
/// Block step for large device reservations.
pub(crate) const LARGE_STEP: usize = 2 << 20;

/// Rounds `n` up to a multiple of `align`.
///
/// Returns `None` on zero input, non-power-of-two alignment, or overflow.
pub(crate) fn round_up(n: usize, align: usize) -> Option<usize> {
    if n == 0 || align == 0 || (align & (align - 1)) != 0 {
        return None;
    }
    let add = align - 1;
    if n > usize::MAX - add {
        return None;
    }
    Some((n + add) & !add)
}

/// Rounds `n` up to the next geometric class.
///
/// Classes are 1x, 1.25x, 1.5x and 1.75x of each power of two, computed
/// with shifts only. Returns `None` on zero input or when no class fits in
/// `usize`.
pub(crate) fn geometric_up(n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let mut base: usize = 1;
    loop {
        let cap = base + (base >> 1) + (base >> 2);
        if cap >= n {
            break;
        }
        if base > usize::MAX / 2 {
            return None;
        }
        base *= 2;
    }
    let quarter = base >> 2;
    let half = base >> 1;
    let mut out = base;
    if n > out {
        out = base + quarter;
    }
    if n > out {
        out = base + half;
    }
    if n > out {
        out = base + half + quarter;
    }
    Some(out)
}

/// Resolves the real reservation size for a theoretical request.
///
/// Returns `None` on zero input or overflow.
pub(crate) fn resolve(requested: usize, is_device: bool, is_pinned: bool) -> Option<usize> {
    if requested == 0 {
        return None;
    }
    if !is_device && !is_pinned {
        return Some(requested);
    }
    if is_pinned {
        if requested <= PINNED_LINEAR_LIMIT {
            round_up(requested, PINNED_SMALL)
        } else {
            geometric_up(requested)
        }
    } else if requested <= DEVICE_LARGE_THRESHOLD {
        round_up(requested, DEVICE_SMALL)
    } else {
        round_up(requested, LARGE_STEP)
    }
}
