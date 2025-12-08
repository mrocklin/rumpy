//! Kernel traits for element-wise operations.
//!
//! Separates "what operation" from "how to traverse memory".
//! Each kernel is a zero-sized type implementing traits for each dtype.
//!
//! Benefits:
//! - Monomorphization: `K::apply(a, b)` compiles to tight code per (kernel, dtype) pair
//! - SIMD lives in loops/, applied uniformly to all operations
//! - Adding a new operation: add kernel struct + impls here
//! - Adding a new dtype: add impls for each kernel

pub mod arithmetic;
pub mod comparison;
pub mod math;

/// Kernel for element-wise binary operations.
pub trait BinaryKernel<T>: Copy {
    fn apply(a: T, b: T) -> T;
}

/// Kernel for element-wise unary operations.
pub trait UnaryKernel<T>: Copy {
    fn apply(v: T) -> T;
}

/// Kernel for reduction operations.
pub trait ReduceKernel<T>: Copy {
    /// Identity element for the reduction.
    fn init() -> T;
    /// Combine accumulator with new value.
    fn combine(acc: T, v: T) -> T;
}

/// Kernel for comparison operations (returns bool).
pub trait CompareKernel<T>: Copy {
    fn apply(a: T, b: T) -> bool;
}
