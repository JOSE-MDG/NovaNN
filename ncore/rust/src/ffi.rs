//! FFI bindings for the storage system.
//!
//! All functions are `extern "C"` and `#[unsafe(no_mangle)]` so they
//! can be called directly from C, C++, or any other language with
//! a C FFI. Each submodule mirrors the corresponding [`ops`]
//! submodule.
//!
//! | Submodule     | FFI Functions                                            |
//! |---------------|----------------------------------------------------------|
//! | [`cpp`]       | `deviceReserve`, `deviceRelease`, `deviceResize`,        |
//! |               | `getDeviceBackend`                                       |
//! | [`reserve`]   | `reserve(size, device, pin_memory, align)`               |
//! | [`lifecycle`] | `retain`, `release`                                      |
//! | [`resize`]    | `resize`                                                 |
//! | [`query`]     | `get_data_from`, `is_valid_handle`, metadata queries     |

pub mod cpp;
pub mod lifecycle;
pub mod query;
pub mod reserve;
pub mod resize;

pub use cpp::{
    DeviceBuffer, DeviceKind, DeviceMemcpyKind, NovaError, NovaStatus, deviceRelease,
    deviceReserve, deviceResize, getDeviceBackend,
};
pub use lifecycle::{release, retain};
pub use query::{
    get_align_from, get_data_from, is_device_memory_handle, is_pinned_handle, is_valid_handle,
};
pub use reserve::reserve;
pub use resize::resize;
