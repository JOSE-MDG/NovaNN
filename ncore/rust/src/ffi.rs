//! FFI bindings for the storage system.
//!
//! All functions are `extern "C"` and `#[unsafe(no_mangle)]` so they
//! can be called directly from C, C++, or any other language with
//! a C FFI. Each submodule mirrors the corresponding [`ops`]
//! submodule.
//!
//! | Submodule     | FFI Functions                                |
//! |---------------|----------------------------------------------|
//! | [`reserve`]   | `reserve`                                    |
//! | [`lifecycle`] | `retain`, `release`                          |
//! | [`resize`]    | `resize`                                     |
//! | [`query`]     | `get_data_from`, `is_valid_handle`           |

pub mod lifecycle;
pub mod query;
pub mod reserve;
pub mod resize;

pub use lifecycle::{release, retain};
pub use query::{get_data_from, is_valid_handle};
pub use reserve::reserve;
pub use resize::resize;
