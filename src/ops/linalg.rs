//! Linear algebra operations.

use crate::array::{DType, RumpyArray};
use faer::prelude::SolverCore;
use faer::{ColRef, MatRef, Side};

/// Copy a faer matrix to a new RumpyArray.
fn faer_to_rumpy(mat: MatRef<'_, f64>) -> RumpyArray {
    let (m, n) = (mat.nrows(), mat.ncols());
    let arr = RumpyArray::zeros(vec![m, n], DType::float64());
    for i in 0..m {
        for j in 0..n {
            unsafe {
                let ptr = arr.data_ptr().add((i * n + j) * 8) as *mut f64;
                *ptr = mat[(i, j)];
            }
        }
    }
    arr
}

/// Copy a faer column vector to a 1D RumpyArray.
fn faer_col_to_rumpy(col: ColRef<'_, f64>) -> RumpyArray {
    let n = col.nrows();
    let arr = RumpyArray::zeros(vec![n], DType::float64());
    for i in 0..n {
        unsafe {
            let ptr = arr.data_ptr().add(i * 8) as *mut f64;
            *ptr = col[i];
        }
    }
    arr
}

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

/// Compute matrix inverse.
pub fn inv(a: &RumpyArray) -> Option<RumpyArray> {
    if a.ndim() != 2 {
        return None;
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return None; // Must be square
    }
    if n == 0 {
        return Some(RumpyArray::zeros(vec![0, 0], DType::float64()));
    }

    let fa = faer::Mat::<f64>::from_fn(n, n, |i, j| a.get_element(&[i, j]));
    let inv = fa.partial_piv_lu().inverse();
    Some(faer_to_rumpy(inv.as_ref()))
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
    let sign = if plu.transposition_count().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };

    Some(det_val * sign)
}

/// Compute Frobenius norm (sqrt of sum of squared elements).
/// Uses vectorized operations: sqrt(sum(x * x)).
pub fn norm(a: &RumpyArray, ord: Option<&str>) -> Option<f64> {
    let ord = ord.unwrap_or("fro");

    match ord {
        "fro" => {
            // Frobenius norm: sqrt(sum(x^2)) using vectorized ops
            let squared = a.binary_op(a, super::BinaryOp::Mul).expect("same shape");
            Some(squared.sum().sqrt())
        }
        _ => None, // Other norms not yet implemented
    }
}

/// QR decomposition: A = QR where Q is orthogonal, R is upper triangular.
///
/// Returns thin (Q, R) where Q is m×k, R is k×n, k = min(m,n).
pub fn qr(a: &RumpyArray) -> Option<(RumpyArray, RumpyArray)> {
    if a.ndim() != 2 {
        return None;
    }
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let fa = faer::Mat::<f64>::from_fn(m, n, |i, j| a.get_element(&[i, j]));

    let decomp = fa.qr();
    let q = faer_to_rumpy(decomp.compute_thin_q().as_ref());
    let r = faer_to_rumpy(decomp.compute_thin_r().as_ref());

    Some((q, r))
}

/// SVD decomposition: A = U @ diag(S) @ V^T.
///
/// Returns thin (U, S, Vt) where U is m×k, S is 1D length k, Vt is k×n.
pub fn svd(a: &RumpyArray) -> Option<(RumpyArray, RumpyArray, RumpyArray)> {
    if a.ndim() != 2 {
        return None;
    }
    let fa = faer::Mat::<f64>::from_fn(a.shape()[0], a.shape()[1], |i, j| a.get_element(&[i, j]));
    let decomp = fa.thin_svd();

    let u = faer_to_rumpy(decomp.u());
    let s = faer_col_to_rumpy(decomp.s_diagonal());
    let vt = faer_to_rumpy(decomp.v().transpose());

    Some((u, s, vt))
}

/// Eigendecomposition of symmetric matrix: A = V @ diag(w) @ V^T.
///
/// Returns (w, V) where w is 1D array of eigenvalues (ascending), V has eigenvectors as columns.
pub fn eigh(a: &RumpyArray) -> Option<(RumpyArray, RumpyArray)> {
    if a.ndim() != 2 {
        return None;
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return None; // Must be square
    }
    if n == 0 {
        return Some((
            RumpyArray::zeros(vec![0], DType::float64()),
            RumpyArray::zeros(vec![0, 0], DType::float64()),
        ));
    }

    let fa = faer::Mat::<f64>::from_fn(n, n, |i, j| a.get_element(&[i, j]));
    let decomp = fa.selfadjoint_eigendecomposition(Side::Lower);

    let w = faer_col_to_rumpy(decomp.s().column_vector());
    let v = faer_to_rumpy(decomp.u());

    Some((w, v))
}

/// Cholesky decomposition: A = L @ L.T for symmetric positive-definite matrix.
///
/// Returns lower triangular matrix L.
pub fn cholesky(a: &RumpyArray) -> Option<RumpyArray> {
    if a.ndim() != 2 {
        return None;
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return None; // Must be square
    }
    if n == 0 {
        return Some(RumpyArray::zeros(vec![0, 0], DType::float64()));
    }

    let fa = faer::Mat::<f64>::from_fn(n, n, |i, j| a.get_element(&[i, j]));
    let chol = fa.cholesky(Side::Lower).ok()?;
    Some(faer_to_rumpy(chol.compute_l().as_ref()))
}

/// Extract diagonal or construct diagonal matrix.
///
/// - 1D input: returns 2D diagonal matrix with input on diagonal
/// - 2D input: returns 1D array of diagonal elements
pub fn diag(a: &RumpyArray) -> Option<RumpyArray> {
    match a.ndim() {
        1 => {
            // Create diagonal matrix from 1D array
            let n = a.shape()[0];
            let result = RumpyArray::zeros(vec![n, n], a.dtype());
            for i in 0..n {
                let val = a.get_element(&[i]);
                unsafe {
                    let ptr = result.data_ptr().add((i * n + i) * 8) as *mut f64;
                    *ptr = val;
                }
            }
            Some(result)
        }
        2 => {
            // Extract diagonal from 2D matrix
            let n = a.shape()[0].min(a.shape()[1]);
            let result = RumpyArray::zeros(vec![n], a.dtype());
            for i in 0..n {
                let val = a.get_element(&[i, i]);
                unsafe {
                    let ptr = result.data_ptr().add(i * 8) as *mut f64;
                    *ptr = val;
                }
            }
            Some(result)
        }
        _ => None,
    }
}
