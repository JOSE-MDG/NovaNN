//! Bindings to the C++ device-memory FFI layer (`csrc/ffi.hpp`).
//!
//! | Path                       | Contents                                         |
//! |----------------------------|--------------------------------------------------|
//! | [`bindings`]               | `extern "C"` declarations                        |
//! | [`crate::status`]          | `NovaStatus`/`NovaError` type mirrors            |

pub mod bindings;
pub use crate::status::{NovaError, NovaStatus};
pub use bindings::{
    DeviceBuffer, DeviceKind, DeviceMemcpyKind, deviceRelease, deviceReserve, deviceResize,
    getDeviceBackend,
};
