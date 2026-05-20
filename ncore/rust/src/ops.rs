//! High-level storage operations.
//!
//! This module is the **operations layer** — the business logic that
//! ties together handles, storage, and the global registry. Each
//! submodule covers a single domain of operations.
//!
//! | Submodule      | Operations                           |
//! |----------------|--------------------------------------|
//! | [`reserve`]    | Allocation and handle creation       |
//! | [`lifecycle`]  | Reference counting (retain / release)|
//! | [`resize`]     | Memory resizing                      |
//! | [`query`]      | Data access and validity checks      |

pub mod lifecycle;
pub mod query;
pub mod reserve;
pub mod resize;

pub use lifecycle::{release_op, retain_op};
pub use query::{get_data_op, is_valid_op};
pub use reserve::reserve_op;
pub use resize::resize_op;
