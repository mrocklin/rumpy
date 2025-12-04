//! Linear algebra operations: trace, det, norm.

use crate::array::RumpyArray;

/// Compute trace of a 2D matrix (sum of diagonal elements).
pub fn trace(a: &RumpyArray) -> Option<f64> {
    if a.ndim() != 2 {
        return None;
    }
    let n = a.shape()[0].min(a.shape()[1]);
    let mut sum = 0.0;
    for i in 0..n {
        sum += a.get_element(&[i, i]);
    }
    Some(sum)
}

/// Compute determinant of a square matrix using LU decomposition.
pub fn det(a: &RumpyArray) -> Option<f64> {
    if a.ndim() != 2 {
        return None;
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return None; // Must be square
    }
    if n == 0 {
        return Some(1.0); // Empty matrix has det = 1
    }

    // Copy to faer Mat
    let fa = faer::Mat::<f64>::from_fn(n, n, |i, j| a.get_element(&[i, j]));

    // LU decomposition with partial pivoting: PA = LU
    let plu = fa.partial_piv_lu();

    // det(A) = det(P^-1) * det(L) * det(U)
    // det(L) = 1 (unit diagonal), det(U) = product of diagonal
    // det(P^-1) = (-1)^(number of transpositions)

    let u = plu.compute_u();
    let mut det_val = 1.0;
    for i in 0..n {
        det_val *= u[(i, i)];
    }

    // Sign from permutation
    let sign = if plu.transposition_count() % 2 == 0 {
        1.0
    } else {
        -1.0
    };

    Some(det_val * sign)
}

/// Compute Frobenius norm (sqrt of sum of squared elements).
pub fn norm(a: &RumpyArray, ord: Option<&str>) -> Option<f64> {
    let ord = ord.unwrap_or("fro");

    match ord {
        "fro" => {
            // Frobenius norm: sqrt(sum of squared elements)
            let mut sum_sq = 0.0;
            let size = a.size();
            let mut indices = vec![0usize; a.ndim()];
            for _ in 0..size {
                let val = a.get_element(&indices);
                sum_sq += val * val;
                crate::array::increment_indices(&mut indices, a.shape());
            }
            Some(sum_sq.sqrt())
        }
        _ => None, // Other norms not yet implemented
    }
}
