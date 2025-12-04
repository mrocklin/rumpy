//! Dot product with numpy-compatible semantics.
//!
//! - scalar × array: element-wise multiplication
//! - 1D × 1D: inner product (scalar result)
//! - 2D × 2D: matrix multiplication
//! - 1D × 2D or 2D × 1D: vector-matrix product

use crate::array::RumpyArray;
use crate::ops::matmul::matmul;

/// Dot product with numpy semantics.
///
/// Returns either a scalar (wrapped as 0D array) or an array depending on inputs.
pub fn dot(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    match (a.ndim(), b.ndim()) {
        (1, 1) => {
            // Inner product: sum of element-wise products
            if a.shape()[0] != b.shape()[0] {
                return None;
            }
            let mut sum = 0.0;
            for i in 0..a.shape()[0] {
                sum += a.get_element(&[i]) * b.get_element(&[i]);
            }
            // Return as 0D array (scalar)
            Some(RumpyArray::full(vec![], sum, a.dtype()))
        }
        (0, _) => {
            // Scalar × array: element-wise
            let scalar = a.get_element(&[]);
            Some(b.scalar_op(scalar, crate::ops::BinaryOp::Mul))
        }
        (_, 0) => {
            // Array × scalar: element-wise
            let scalar = b.get_element(&[]);
            Some(a.scalar_op(scalar, crate::ops::BinaryOp::Mul))
        }
        _ => {
            // 2D cases (and higher): use matmul
            matmul(a, b)
        }
    }
}
