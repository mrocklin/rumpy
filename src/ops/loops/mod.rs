//! Loop strategies for memory traversal.
//!
//! Separates "how to traverse memory" from "what operation".
//! SIMD optimizations live here, applied uniformly to all operations.
//!
//! Layout detection happens once at dispatch time, not embedded in every kernel.

pub mod contiguous;
pub mod strided;

pub use contiguous::{map_binary, map_unary, map_compare, reduce};
pub use strided::{map_binary_strided, map_unary_strided, map_compare_strided, reduce_strided};
