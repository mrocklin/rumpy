//! Array method implementations split by category.
//!
//! These modules extend `RumpyArray` with methods organized by functionality.

pub mod cumulative;
pub mod logical;
pub mod reductions;
pub mod sorting;
pub mod unary;

// Re-export lexsort as a module-level function
pub use sorting::lexsort;
