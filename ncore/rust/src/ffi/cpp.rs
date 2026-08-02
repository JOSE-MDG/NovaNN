//! Bindings to the C++ device-memory FFI layer (`csrc/ffi.hpp`).
//!
//! | Path                       | Contents                                         |
//! |----------------------------|--------------------------------------------------|
//! | [`bindings`]               | `extern "C"` declarations, type mirrors           |

pub mod bindings;

pub use bindings::{
    DeviceBuffer, DeviceKind, DeviceMemcpyKind, NovaError, NovaStatus, deviceRelease,
    deviceReserve, deviceResize, getDeviceBackend,
};
