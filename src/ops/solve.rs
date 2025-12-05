//! Linear system solver.
//!
//! Solves Ax = b for x using LU decomposition with partial pivoting.

use crate::array::RumpyArray;
use faer::prelude::SpSolver;

/// Solve linear system Ax = b.
///
/// Uses LU decomposition with partial pivoting.
/// A must be square and invertible.
///
/// # Arguments
/// * `a` - Square matrix (n, n)
/// * `b` - Right-hand side, either (n,) or (n, m)
///
/// # Returns
/// Solution x with same shape as b
pub fn solve(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    // Validate A is 2D square
    if a.ndim() != 2 {
        return None;
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return None;
    }

    // Handle 1D b by expanding to column vector
    let (b_expanded, squeeze) = if b.ndim() == 1 {
        if b.shape()[0] != n {
            return None;
        }
        (b.expand_dims(1)?, true)
    } else if b.ndim() == 2 {
        if b.shape()[0] != n {
            return None;
        }
        (b.clone(), false)
    } else {
        return None;
    };

    let m = b_expanded.shape()[1]; // number of right-hand sides

    // Copy A and b into faer Mat
    let fa = faer::Mat::<f64>::from_fn(n, n, |i, j| a.get_element(&[i, j]));
    let fb = faer::Mat::<f64>::from_fn(n, m, |i, j| b_expanded.get_element(&[i, j]));

    // Solve using LU with partial pivoting
    let plu = fa.partial_piv_lu();
    let fx = plu.solve(&fb);

    // Copy result to output
    let result = RumpyArray::zeros(vec![n, m], a.dtype());
    let result_ptr = result.data_ptr() as *mut u8;
    let r_stride_0 = result.strides()[0];
    let r_stride_1 = result.strides()[1];

    for i in 0..n {
        for j in 0..m {
            let byte_offset = (i as isize) * r_stride_0 + (j as isize) * r_stride_1;
            unsafe {
                let dst = result_ptr.offset(byte_offset);
                *(dst as *mut f64) = fx[(i, j)];
            }
        }
    }

    // Squeeze back to 1D if input was 1D
    if squeeze {
        Some(result.squeeze())
    } else {
        Some(result)
    }
}
