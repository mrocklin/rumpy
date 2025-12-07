//! Linear algebra operations.

use crate::array::{DType, RumpyArray};
use crate::array::dtype::DTypeKind;
use faer::prelude::SolverCore;
use faer::{ColRef, MatRef, Side};

/// Try to create a zero-copy faer MatRef from a RumpyArray.
/// Returns None if the array is not contiguous float64.
fn try_as_faer_mat(a: &RumpyArray) -> Option<MatRef<'_, f64>> {
    if a.ndim() != 2 || a.dtype().kind() != DTypeKind::Float64 {
        return None;
    }
    if !a.is_c_contiguous() {
        return None;
    }
    let (m, n) = (a.shape()[0], a.shape()[1]);
    if m == 0 || n == 0 {
        return None;
    }
    unsafe {
        Some(faer::mat::from_raw_parts::<f64, usize, usize>(
            a.data_ptr() as *const f64,
            m, n,
            n as isize, 1,  // row_stride = n, col_stride = 1 for C-contiguous
        ))
    }
}

/// Create a faer Mat by copying data (fallback for non-contiguous arrays).
fn copy_to_faer_mat(a: &RumpyArray) -> faer::Mat<f64> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    faer::Mat::<f64>::from_fn(m, n, |i, j| a.get_element(&[i, j]))
}

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

    // Try zero-copy path first
    if let Some(fa) = try_as_faer_mat(a) {
        let inv = fa.partial_piv_lu().inverse();
        return Some(faer_to_rumpy(inv.as_ref()));
    }
    // Fallback: copy data
    let fa = copy_to_faer_mat(a);
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

    // LU decomposition with partial pivoting: PA = LU
    let plu = if let Some(fa) = try_as_faer_mat(a) {
        fa.partial_piv_lu()
    } else {
        copy_to_faer_mat(a).partial_piv_lu()
    };

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

    let decomp = if let Some(fa) = try_as_faer_mat(a) {
        fa.qr()
    } else {
        copy_to_faer_mat(a).qr()
    };
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

    let decomp = if let Some(fa) = try_as_faer_mat(a) {
        fa.thin_svd()
    } else {
        copy_to_faer_mat(a).thin_svd()
    };

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

    let decomp = if let Some(fa) = try_as_faer_mat(a) {
        fa.selfadjoint_eigendecomposition(Side::Lower)
    } else {
        copy_to_faer_mat(a).selfadjoint_eigendecomposition(Side::Lower)
    };

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

    let chol = if let Some(fa) = try_as_faer_mat(a) {
        fa.cholesky(Side::Lower).ok()?
    } else {
        copy_to_faer_mat(a).cholesky(Side::Lower).ok()?
    };
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

/// Sign and log of determinant (for numerical stability with large determinants).
/// Returns (sign, logabsdet) where det(A) = sign * exp(logabsdet).
pub fn slogdet(a: &RumpyArray) -> Option<(f64, f64)> {
    if a.ndim() != 2 {
        return None;
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return None; // Must be square
    }
    if n == 0 {
        return Some((1.0, f64::NEG_INFINITY)); // Empty matrix: det = 1, log(1) = 0, but we return -inf for log(0) product convention
    }

    let plu = if let Some(fa) = try_as_faer_mat(a) {
        fa.partial_piv_lu()
    } else {
        copy_to_faer_mat(a).partial_piv_lu()
    };
    let u = plu.compute_u();

    // Compute sign and log of absolute determinant
    let mut sign = if plu.transposition_count().is_multiple_of(2) { 1.0 } else { -1.0 };
    let mut logabsdet = 0.0;

    for i in 0..n {
        let diag_val = u[(i, i)];
        if diag_val == 0.0 {
            return Some((0.0, f64::NEG_INFINITY));
        }
        if diag_val < 0.0 {
            sign = -sign;
        }
        logabsdet += diag_val.abs().ln();
    }

    Some((sign, logabsdet))
}

/// Compute condition number of a matrix.
/// cond(A) = ||A|| * ||A^-1|| = sigma_max / sigma_min (using SVD).
pub fn cond(a: &RumpyArray, p: Option<&str>) -> Option<f64> {
    if a.ndim() != 2 {
        return None;
    }

    let p = p.unwrap_or("2");

    match p {
        "2" | "-2" => {
            // Use SVD: cond_2 = sigma_max / sigma_min
            let sv = if let Some(fa) = try_as_faer_mat(a) {
                fa.singular_values()
            } else {
                copy_to_faer_mat(a).singular_values()
            };

            if sv.is_empty() {
                return Some(0.0);
            }

            let sigma_max = sv[0];
            let sigma_min = sv[sv.len() - 1];

            if sigma_min == 0.0 {
                return Some(f64::INFINITY);
            }

            if p == "-2" {
                Some(sigma_min / sigma_max)
            } else {
                Some(sigma_max / sigma_min)
            }
        }
        _ => None, // Other norms not yet implemented
    }
}

/// Compute matrix rank using SVD.
/// Rank is count of singular values greater than tolerance.
pub fn matrix_rank(a: &RumpyArray, tol: Option<f64>) -> Option<usize> {
    if a.ndim() != 2 {
        return None;
    }

    let (m, n) = (a.shape()[0], a.shape()[1]);
    if m == 0 || n == 0 {
        return Some(0);
    }

    let sv = if let Some(fa) = try_as_faer_mat(a) {
        fa.singular_values()
    } else {
        copy_to_faer_mat(a).singular_values()
    };

    if sv.is_empty() {
        return Some(0);
    }

    // Default tolerance: max(m,n) * eps * sigma_max
    let sigma_max = sv[0];
    let default_tol = (m.max(n) as f64) * f64::EPSILON * sigma_max;
    let tol = tol.unwrap_or(default_tol);

    let mut rank = 0;
    for s in &sv {
        if *s > tol {
            rank += 1;
        }
    }

    Some(rank)
}

/// Moore-Penrose pseudo-inverse using SVD.
/// pinv(A) = V @ diag(1/S) @ U^T (for non-zero singular values).
pub fn pinv(a: &RumpyArray, rcond: Option<f64>) -> Option<RumpyArray> {
    if a.ndim() != 2 {
        return None;
    }

    let (m, n) = (a.shape()[0], a.shape()[1]);
    if m == 0 || n == 0 {
        return Some(RumpyArray::zeros(vec![n, m], DType::float64()));
    }

    let decomp = if let Some(fa) = try_as_faer_mat(a) {
        fa.thin_svd()
    } else {
        copy_to_faer_mat(a).thin_svd()
    };

    let u = decomp.u();
    let s = decomp.s_diagonal();
    let vt = decomp.v().transpose();

    // Cutoff for small singular values
    let rcond = rcond.unwrap_or(f64::EPSILON * (m.max(n) as f64));
    let cutoff = rcond * s[0].abs(); // threshold = rcond * sigma_max

    // Build pseudo-inverse: V @ diag(1/S) @ U^T
    // Result is n x m
    let result = RumpyArray::zeros(vec![n, m], DType::float64());

    let k = s.nrows(); // number of singular values

    for i in 0..n {
        for j in 0..m {
            let mut val = 0.0;
            for l in 0..k {
                let sv = s[l];
                if sv.abs() > cutoff {
                    // pinv[i,j] = sum_l (V[i,l] * (1/S[l]) * U[j,l])
                    // Note: vt is V^T, so V[i,l] = vt[l,i]
                    val += vt[(l, i)] * (1.0 / sv) * u[(j, l)];
                }
            }
            unsafe {
                let ptr = result.data_ptr().add((i * m + j) * 8) as *mut f64;
                *ptr = val;
            }
        }
    }

    Some(result)
}

/// Least-squares solution to Ax = b.
/// Returns (x, residuals, rank, s) where:
/// - x is the solution
/// - residuals is sum of squared residuals (only if m > n and rank == n)
/// - rank is the matrix rank
/// - s is the singular values
pub fn lstsq(a: &RumpyArray, b: &RumpyArray, rcond: Option<f64>) -> Option<(RumpyArray, RumpyArray, usize, RumpyArray)> {
    if a.ndim() != 2 {
        return None;
    }

    let (m, n) = (a.shape()[0], a.shape()[1]);

    // b can be 1D (m,) or 2D (m, k)
    let b_is_1d = b.ndim() == 1;
    let b_cols = if b_is_1d {
        if b.shape()[0] != m {
            return None;
        }
        1
    } else if b.ndim() == 2 {
        if b.shape()[0] != m {
            return None;
        }
        b.shape()[1]
    } else {
        return None;
    };

    if m == 0 || n == 0 {
        let x = RumpyArray::zeros(if b_is_1d { vec![n] } else { vec![n, b_cols] }, DType::float64());
        let residuals = RumpyArray::zeros(vec![0], DType::float64());
        let s = RumpyArray::zeros(vec![0], DType::float64());
        return Some((x, residuals, 0, s));
    }

    let decomp = if let Some(fa) = try_as_faer_mat(a) {
        fa.thin_svd()
    } else {
        copy_to_faer_mat(a).thin_svd()
    };

    let u = decomp.u();
    let s_vals = decomp.s_diagonal();
    let v = decomp.v();

    // Determine rank based on rcond
    let rcond = rcond.unwrap_or(-1.0); // -1 means use machine precision
    let rcond = if rcond < 0.0 { f64::EPSILON * (m.max(n) as f64) } else { rcond };
    let cutoff = rcond * s_vals[0].abs();

    let mut rank = 0;
    for i in 0..s_vals.nrows() {
        if s_vals[i].abs() > cutoff {
            rank += 1;
        }
    }

    // Copy singular values to output
    let s = RumpyArray::zeros(vec![s_vals.nrows()], DType::float64());
    for i in 0..s_vals.nrows() {
        unsafe {
            let ptr = s.data_ptr().add(i * 8) as *mut f64;
            *ptr = s_vals[i];
        }
    }

    // Solve for each column of b
    let x = RumpyArray::zeros(if b_is_1d { vec![n] } else { vec![n, b_cols] }, DType::float64());

    for col in 0..b_cols {
        // Get b column
        let mut b_col = vec![0.0; m];
        for i in 0..m {
            b_col[i] = if b_is_1d {
                b.get_element(&[i])
            } else {
                b.get_element(&[i, col])
            };
        }

        // x = V @ (S^-1 @ (U^T @ b)) for non-zero singular values
        // Step 1: c = U^T @ b
        let mut c = vec![0.0; s_vals.nrows()];
        for i in 0..s_vals.nrows() {
            for j in 0..m {
                c[i] += u[(j, i)] * b_col[j];
            }
        }

        // Step 2: c = S^-1 @ c (only for non-zero singular values)
        for i in 0..s_vals.nrows() {
            if s_vals[i].abs() > cutoff {
                c[i] /= s_vals[i];
            } else {
                c[i] = 0.0;
            }
        }

        // Step 3: x = V @ c
        for i in 0..n {
            let mut val = 0.0;
            for j in 0..s_vals.nrows() {
                val += v[(i, j)] * c[j];
            }
            unsafe {
                let idx = if b_is_1d { i } else { i * b_cols + col };
                let ptr = x.data_ptr().add(idx * 8) as *mut f64;
                *ptr = val;
            }
        }
    }

    // Compute residuals only if m > n and rank == n
    let residuals = if m > n && rank == n {
        let res = RumpyArray::zeros(vec![b_cols], DType::float64());
        for col in 0..b_cols {
            let mut sum_sq = 0.0;
            for i in 0..m {
                let mut ax_i = 0.0;
                for j in 0..n {
                    let x_j = if b_is_1d {
                        x.get_element(&[j])
                    } else {
                        x.get_element(&[j, col])
                    };
                    ax_i += a.get_element(&[i, j]) * x_j;
                }
                let b_i = if b_is_1d {
                    b.get_element(&[i])
                } else {
                    b.get_element(&[i, col])
                };
                sum_sq += (ax_i - b_i).powi(2);
            }
            unsafe {
                let ptr = res.data_ptr().add(col * 8) as *mut f64;
                *ptr = sum_sq;
            }
        }
        res
    } else {
        RumpyArray::zeros(vec![0], DType::float64())
    };

    Some((x, residuals, rank, s))
}

/// Eigenvalues only (for general non-symmetric matrices).
/// Returns complex eigenvalues as a complex128 array.
pub fn eigvals(a: &RumpyArray) -> Option<RumpyArray> {
    if a.ndim() != 2 {
        return None;
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return None;
    }
    if n == 0 {
        return Some(RumpyArray::zeros(vec![0], DType::complex128()));
    }

    let eigenvalues = if let Some(fa) = try_as_faer_mat(a) {
        fa.eigenvalues::<faer::complex_native::c64>()
    } else {
        copy_to_faer_mat(a).eigenvalues::<faer::complex_native::c64>()
    };

    // Create complex128 result array
    let result = RumpyArray::zeros(vec![n], DType::complex128());
    for i in 0..n {
        let ev = eigenvalues[i];
        unsafe {
            let ptr = result.data_ptr().add(i * 16) as *mut f64;
            *ptr = ev.re;
            *ptr.add(1) = ev.im;
        }
    }

    Some(result)
}

/// General eigendecomposition (for non-symmetric matrices).
/// Returns (w, V) where w is complex eigenvalues and V is complex eigenvectors.
pub fn eig(a: &RumpyArray) -> Option<(RumpyArray, RumpyArray)> {
    if a.ndim() != 2 {
        return None;
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return None;
    }
    if n == 0 {
        return Some((
            RumpyArray::zeros(vec![0], DType::complex128()),
            RumpyArray::zeros(vec![0, 0], DType::complex128()),
        ));
    }

    let decomp = if let Some(fa) = try_as_faer_mat(a) {
        fa.eigendecomposition::<faer::complex_native::c64>()
    } else {
        copy_to_faer_mat(a).eigendecomposition::<faer::complex_native::c64>()
    };

    let eigenvalues = decomp.s().column_vector();
    let eigenvectors = decomp.u();

    // Create complex128 eigenvalues
    let w = RumpyArray::zeros(vec![n], DType::complex128());
    for i in 0..n {
        let ev = eigenvalues[i];
        unsafe {
            let ptr = w.data_ptr().add(i * 16) as *mut f64;
            *ptr = ev.re;
            *ptr.add(1) = ev.im;
        }
    }

    // Create complex128 eigenvectors (columns are eigenvectors)
    let v = RumpyArray::zeros(vec![n, n], DType::complex128());
    for i in 0..n {
        for j in 0..n {
            let val = eigenvectors[(i, j)];
            unsafe {
                let ptr = v.data_ptr().add((i * n + j) * 16) as *mut f64;
                *ptr = val.re;
                *ptr.add(1) = val.im;
            }
        }
    }

    Some((w, v))
}

/// Vector dot product (conjugate first argument for complex).
/// For 1D arrays: sum(conj(a) * b)
/// For nD arrays: dot product over last axes (flattened).
pub fn vdot(a: &RumpyArray, b: &RumpyArray) -> Option<f64> {
    if a.size() != b.size() {
        return None;
    }

    let n = a.size();
    if n == 0 {
        return Some(0.0);
    }

    // Fast path for contiguous float64 arrays using faer's optimized matmul
    // Compute as row × column: [1,n] @ [n,1] -> [1,1]
    if a.is_c_contiguous() && b.is_c_contiguous()
        && a.dtype().kind() == DTypeKind::Float64
        && b.dtype().kind() == DTypeKind::Float64
    {
        unsafe {
            // Treat 'a' as row vector [1, n]
            let fa = faer::mat::from_raw_parts::<f64, usize, usize>(
                a.data_ptr() as *const f64,
                1, n,  // 1 row, n cols
                n as isize, 1,  // row_stride (not used), col_stride = 1
            );
            // Treat 'b' as column vector [n, 1]
            let fb = faer::mat::from_raw_parts::<f64, usize, usize>(
                b.data_ptr() as *const f64,
                n, 1,  // n rows, 1 col
                1, 1,  // row_stride = 1, col_stride (not used)
            );
            // Result is [1, 1]
            let mut result = 0.0f64;
            let mut fc = faer::mat::from_raw_parts_mut::<f64, usize, usize>(
                &mut result as *mut f64,
                1, 1,
                1, 1,
            );
            faer::linalg::matmul::matmul(
                fc.as_mut(),
                fa.as_ref(),
                fb.as_ref(),
                None,
                1.0,
                faer::Parallelism::None,
            );
            return Some(result);
        }
    }

    // General path for non-contiguous or other dtypes
    let a_ptr = a.data_ptr();
    let b_ptr = b.data_ptr();
    let a_dtype = a.dtype();
    let b_dtype = b.dtype();

    let sum = a.iter_offsets().zip(b.iter_offsets()).fold(0.0, |acc, (a_off, b_off)| {
        let a_val = unsafe { a_dtype.ops().read_f64(a_ptr, a_off).unwrap_or(0.0) };
        let b_val = unsafe { b_dtype.ops().read_f64(b_ptr, b_off).unwrap_or(0.0) };
        acc + a_val * b_val
    });

    Some(sum)
}

/// Kronecker product of two arrays.
pub fn kron(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    if a.ndim() < 1 || b.ndim() < 1 {
        return None;
    }

    // Broadcast to same ndim (prepend 1s to smaller)
    let max_ndim = a.ndim().max(b.ndim());
    let mut a_shape = vec![1usize; max_ndim];
    let mut b_shape = vec![1usize; max_ndim];

    for i in 0..a.ndim() {
        a_shape[max_ndim - a.ndim() + i] = a.shape()[i];
    }
    for i in 0..b.ndim() {
        b_shape[max_ndim - b.ndim() + i] = b.shape()[i];
    }

    // Result shape is element-wise product
    let mut out_shape = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        out_shape.push(a_shape[i] * b_shape[i]);
    }

    let result = RumpyArray::zeros(out_shape.clone(), DType::float64());

    // For 2D case (most common)
    if max_ndim == 2 {
        let (m, n) = (a_shape[0], a_shape[1]);
        let (p, q) = (b_shape[0], b_shape[1]);
        let out_cols = out_shape[1];

        // Fast path for contiguous float64 arrays
        if a.is_c_contiguous() && b.is_c_contiguous()
            && a.dtype().kind() == DTypeKind::Float64
            && b.dtype().kind() == DTypeKind::Float64
        {
            let a_ptr = a.data_ptr() as *const f64;
            let b_ptr = b.data_ptr() as *const f64;
            let out_ptr = result.data_ptr() as *mut f64;

            for i in 0..m {
                for j in 0..n {
                    let a_val = unsafe { *a_ptr.add(i * n + j) };
                    for k in 0..p {
                        for l in 0..q {
                            let b_val = unsafe { *b_ptr.add(k * q + l) };
                            let out_i = i * p + k;
                            let out_j = j * q + l;
                            unsafe {
                                *out_ptr.add(out_i * out_cols + out_j) = a_val * b_val;
                            }
                        }
                    }
                }
            }
        } else {
            // General path
            for i in 0..m {
                for j in 0..n {
                    let a_val = a.get_element(&[i, j]);
                    for k in 0..p {
                        for l in 0..q {
                            let b_val = b.get_element(&[k, l]);
                            let out_i = i * p + k;
                            let out_j = j * q + l;
                            unsafe {
                                let ptr = result.data_ptr().add((out_i * out_cols + out_j) * 8) as *mut f64;
                                *ptr = a_val * b_val;
                            }
                        }
                    }
                }
            }
        }
    } else if max_ndim == 1 {
        // 1D case
        let m = a_shape[0];
        let p = b_shape[0];

        // Fast path for contiguous float64
        if a.is_c_contiguous() && b.is_c_contiguous()
            && a.dtype().kind() == DTypeKind::Float64
            && b.dtype().kind() == DTypeKind::Float64
        {
            let a_ptr = a.data_ptr() as *const f64;
            let b_ptr = b.data_ptr() as *const f64;
            let out_ptr = result.data_ptr() as *mut f64;

            for i in 0..m {
                let a_val = unsafe { *a_ptr.add(i) };
                for k in 0..p {
                    let b_val = unsafe { *b_ptr.add(k) };
                    unsafe {
                        *out_ptr.add(i * p + k) = a_val * b_val;
                    }
                }
            }
        } else {
            for i in 0..m {
                let a_val = a.get_element(&[i]);
                for k in 0..p {
                    let b_val = b.get_element(&[k]);
                    unsafe {
                        let ptr = result.data_ptr().add((i * p + k) * 8) as *mut f64;
                        *ptr = a_val * b_val;
                    }
                }
            }
        }
    } else {
        // General nD case - not yet implemented
        return None;
    }

    Some(result)
}

/// Cross product of two 3D vectors.
pub fn cross(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    // Simplified: only handle 1D arrays of length 3
    if a.ndim() != 1 || b.ndim() != 1 {
        return None;
    }
    if a.shape()[0] != 3 || b.shape()[0] != 3 {
        return None;
    }

    let a0 = a.get_element(&[0]);
    let a1 = a.get_element(&[1]);
    let a2 = a.get_element(&[2]);
    let b0 = b.get_element(&[0]);
    let b1 = b.get_element(&[1]);
    let b2 = b.get_element(&[2]);

    let result = RumpyArray::zeros(vec![3], DType::float64());
    unsafe {
        let ptr = result.data_ptr() as *mut f64;
        *ptr = a1 * b2 - a2 * b1;
        *ptr.add(1) = a2 * b0 - a0 * b2;
        *ptr.add(2) = a0 * b1 - a1 * b0;
    }

    Some(result)
}

/// Tensor dot product over specified axes.
/// axes can be:
/// - A single integer n: sum over last n axes of a and first n axes of b
/// - A pair (a_axes, b_axes): sum over these specific axes
pub fn tensordot(a: &RumpyArray, b: &RumpyArray, axes: (Vec<usize>, Vec<usize>)) -> Option<RumpyArray> {
    let (a_axes, b_axes) = axes;

    if a_axes.len() != b_axes.len() {
        return None;
    }

    // Validate axes
    for &ax in &a_axes {
        if ax >= a.ndim() {
            return None;
        }
    }
    for &ax in &b_axes {
        if ax >= b.ndim() {
            return None;
        }
    }

    // Check that contracted dimensions match
    for i in 0..a_axes.len() {
        if a.shape()[a_axes[i]] != b.shape()[b_axes[i]] {
            return None;
        }
    }

    // Simple case: tensordot with axes=1 (matmul-like)
    if a.ndim() == 2 && b.ndim() == 2 && a_axes == vec![1] && b_axes == vec![0] {
        // This is just matrix multiplication
        return crate::ops::matmul::matmul(a, b);
    }

    // General case: compute result shape and contract
    // Result shape = [a_shape without a_axes] + [b_shape without b_axes]
    let mut result_shape = Vec::new();
    for i in 0..a.ndim() {
        if !a_axes.contains(&i) {
            result_shape.push(a.shape()[i]);
        }
    }
    for i in 0..b.ndim() {
        if !b_axes.contains(&i) {
            result_shape.push(b.shape()[i]);
        }
    }

    if result_shape.is_empty() {
        result_shape.push(1); // Scalar result
    }

    let result = RumpyArray::zeros(result_shape.clone(), DType::float64());

    // For now, only support up to 2D arrays
    if a.ndim() > 2 || b.ndim() > 2 {
        return None;
    }

    // 1D cases
    if a.ndim() == 1 && b.ndim() == 1 {
        // Inner product
        let mut sum = 0.0;
        for i in 0..a.shape()[0] {
            sum += a.get_element(&[i]) * b.get_element(&[i]);
        }
        unsafe {
            let ptr = result.data_ptr() as *mut f64;
            *ptr = sum;
        }
        return Some(result);
    }

    // 2D x 1D or 1D x 2D
    if a.ndim() == 2 && b.ndim() == 1 {
        // axes=([1], [0]) means matrix-vector product
        if a_axes == vec![1] && b_axes == vec![0] {
            for i in 0..a.shape()[0] {
                let mut sum = 0.0;
                for j in 0..a.shape()[1] {
                    sum += a.get_element(&[i, j]) * b.get_element(&[j]);
                }
                unsafe {
                    let ptr = result.data_ptr().add(i * 8) as *mut f64;
                    *ptr = sum;
                }
            }
            return Some(result);
        }
    }

    if a.ndim() == 1 && b.ndim() == 2 {
        // axes=([0], [0]) means vector-matrix product
        if a_axes == vec![0] && b_axes == vec![0] {
            for j in 0..b.shape()[1] {
                let mut sum = 0.0;
                for i in 0..a.shape()[0] {
                    sum += a.get_element(&[i]) * b.get_element(&[i, j]);
                }
                unsafe {
                    let ptr = result.data_ptr().add(j * 8) as *mut f64;
                    *ptr = sum;
                }
            }
            return Some(result);
        }
    }

    None // Unsupported case
}
