//! Float dtype implementations (float32, float64).
//! Float16 has special handling due to the half crate.

use super::macros::impl_float_dtype;
use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};

impl_float_dtype!(Float64Ops, f64, 8, Float64, "float64", "<f8", "d", 100);
impl_float_dtype!(Float32Ops, f32, 4, Float32, "float32", "<f4", "f", 90);
