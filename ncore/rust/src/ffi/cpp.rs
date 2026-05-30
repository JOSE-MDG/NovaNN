//! Bindings to the C++ device-memory FFI layer (`ncore/rust/csrc/ffi.hpp`).
//!
//! | Path                       | Contents                                         |
//! |----------------------------|--------------------------------------------------|
//! | [`bindings`]               | `extern "C"` declarations, type mirrors           |

pub mod bindings;

pub use bindings::{
    DeviceBuffer, DeviceKind, DeviceMemcpyKind, DeviceStatus, device_memcpy, device_release,
    device_reserve, device_resize, get_device_backend,
};
