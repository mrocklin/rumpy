//! Polynomial operations: polyfit, polyval, polyder, polyint, roots.
//!
//! Follows NumPy's polynomial coefficient convention: highest degree first.
//! E.g., [2, 3, 1] represents 2x² + 3x + 1.

use crate::array::{DType, RumpyArray};
use crate::ops::linalg;

/// Collect array elements to Vec<f64>, handling any dtype and striding.
fn array_to_vec(arr: &RumpyArray) -> Vec<f64> {
    let size = arr.size();
    if size == 0 {
        return Vec::new();
    }

    let ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();
    let mut values = Vec::with_capacity(size);
    for offset in arr.iter_offsets() {
        values.push(unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0));
    }
    values
}

/// Evaluate polynomial at given points.
///
/// Coefficients are in descending degree order (highest first).
/// Uses Horner's method for numerical stability.
pub fn polyval(p: &RumpyArray, x: &RumpyArray) -> RumpyArray {
    let coeffs = array_to_vec(p);  // Coefficients are typically small
    let n = coeffs.len();

    if n == 0 {
        return RumpyArray::zeros(x.shape().to_vec(), DType::float64());
    }

    let size = x.size();
    let mut result = Vec::with_capacity(size);

    // Iterate directly over x without copying
    let ptr = x.data_ptr();
    let dtype = x.dtype();
    let ops = dtype.ops();

    // Horner's method: p(x) = (...((c[0]*x + c[1])*x + c[2])*x + ... + c[n-1])
    for offset in x.iter_offsets() {
        let xv = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
        let mut acc = coeffs[0];
        for &c in &coeffs[1..] {
            acc = acc * xv + c;
        }
        result.push(acc);
    }

    let arr = RumpyArray::from_vec(result, DType::float64());
    arr.reshape(x.shape().to_vec()).unwrap_or(arr)
}

/// Compute polynomial derivative.
///
/// Returns coefficients of d^m/dx^m polynomial.
/// For a polynomial of degree n, derivative has degree n-1.
pub fn polyder(p: &RumpyArray, m: usize) -> RumpyArray {
    if m == 0 {
        return p.copy();
    }

    let mut coeffs = array_to_vec(p);

    for _ in 0..m {
        if coeffs.len() <= 1 {
            // Derivative of constant or empty is empty
            return RumpyArray::from_vec(Vec::new(), DType::float64());
        }

        let n = coeffs.len();
        // d/dx (c[i] * x^(n-1-i)) = c[i] * (n-1-i) * x^(n-2-i)
        let mut new_coeffs = Vec::with_capacity(n - 1);
        for (i, &c) in coeffs.iter().take(n - 1).enumerate() {
            let power = (n - 1 - i) as f64;
            new_coeffs.push(c * power);
        }
        coeffs = new_coeffs;
    }

    RumpyArray::from_vec(coeffs, DType::float64())
}

/// Compute polynomial integral.
///
/// Returns coefficients of antiderivative. Integration constant k is appended.
/// For multiple integrals, m specifies order and k can be array of constants.
pub fn polyint(p: &RumpyArray, m: usize, k: Option<&RumpyArray>) -> RumpyArray {
    if m == 0 {
        return p.copy();
    }

    let mut coeffs = array_to_vec(p);
    let k_values = k.map(array_to_vec).unwrap_or_default();

    for i in 0..m {
        let n = coeffs.len();
        // integral of c[i] * x^(n-1-i) = c[i]/(n-i) * x^(n-i)
        let mut new_coeffs = Vec::with_capacity(n + 1);
        for (j, &c) in coeffs.iter().enumerate() {
            let power = (n - j) as f64;
            new_coeffs.push(c / power);
        }
        // Append integration constant
        let constant = k_values.get(m - 1 - i).copied().unwrap_or(0.0);
        new_coeffs.push(constant);
        coeffs = new_coeffs;
    }

    RumpyArray::from_vec(coeffs, DType::float64())
}

/// Least squares polynomial fit.
///
/// Fit polynomial of degree `deg` to data points (x, y).
/// Returns coefficients in descending order (highest degree first).
///
/// Uses the Vandermonde matrix approach:
/// V * c = y, where V[i,j] = x[i]^(deg-j)
/// Solved using QR decomposition via linalg.
pub fn polyfit(x: &RumpyArray, y: &RumpyArray, deg: usize, w: Option<&RumpyArray>) -> Option<RumpyArray> {
    let x_vals = array_to_vec(x);
    let y_vals = array_to_vec(y);
    let n = x_vals.len();

    if n == 0 || n != y_vals.len() || deg >= n {
        return None;
    }

    // Build Vandermonde matrix: V[i,j] = x[i]^(deg-j)
    let cols = deg + 1;
    let mut vander_data = Vec::with_capacity(n * cols);

    for &xv in &x_vals {
        // Powers from deg down to 0
        let mut power = 1.0;
        let mut row = vec![0.0; cols];
        row[cols - 1] = 1.0;  // x^0 = 1
        for j in (0..cols - 1).rev() {
            power *= xv;
            row[j] = power;
        }
        vander_data.extend(row);
    }

    let vander = RumpyArray::from_vec(vander_data, DType::float64())
        .reshape(vec![n, cols])?;

    let y_col = RumpyArray::from_vec(y_vals.clone(), DType::float64())
        .reshape(vec![n, 1])?;

    // Apply weights if provided
    let (vander_w, y_w) = if let Some(weights) = w {
        let w_vals = array_to_vec(weights);
        if w_vals.len() != n {
            return None;
        }

        // Weight by sqrt(w): W*V and W*y where W = diag(sqrt(w))
        let mut v_data = Vec::with_capacity(n * cols);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let sw = w_vals[i].sqrt();
            for j in 0..cols {
                v_data.push(vander.get_element(&[i, j]) * sw);
            }
            y_data.push(y_vals[i] * sw);
        }

        let vw = RumpyArray::from_vec(v_data, DType::float64()).reshape(vec![n, cols])?;
        let yw = RumpyArray::from_vec(y_data, DType::float64()).reshape(vec![n, 1])?;
        (vw, yw)
    } else {
        (vander, y_col)
    };

    // Solve via least squares: (V^T V) c = V^T y
    // Using QR: V = QR, then R c = Q^T y
    let (q, r) = linalg::qr(&vander_w)?;
    let qt = q.transpose();
    let qty = crate::ops::matmul::matmul(&qt, &y_w)?;

    // Back-substitute: R * c = qty
    // R is upper triangular
    let mut coeffs = vec![0.0; cols];
    for i in (0..cols).rev() {
        let mut sum = qty.get_element(&[i, 0]);
        for j in (i + 1)..cols {
            sum -= r.get_element(&[i, j]) * coeffs[j];
        }
        let r_ii = r.get_element(&[i, i]);
        if r_ii.abs() < 1e-15 {
            // Near-singular, use pseudoinverse behavior
            coeffs[i] = 0.0;
        } else {
            coeffs[i] = sum / r_ii;
        }
    }

    Some(RumpyArray::from_vec(coeffs, DType::float64()))
}

/// Find roots of polynomial.
///
/// Uses direct formulas for low-degree polynomials and companion matrix
/// eigenvalue computation for higher degrees.
pub fn roots(p: &RumpyArray) -> Option<RumpyArray> {
    let mut coeffs = array_to_vec(p);

    // Remove leading zeros
    while !coeffs.is_empty() && coeffs[0].abs() < 1e-15 {
        coeffs.remove(0);
    }

    let n = coeffs.len();
    if n <= 1 {
        // Constant or empty - no roots
        return Some(RumpyArray::from_vec(Vec::new(), DType::float64()));
    }

    let degree = n - 1;

    // Normalize by leading coefficient
    let lead = coeffs[0];
    for c in coeffs.iter_mut() {
        *c /= lead;
    }

    // For low-degree polynomials, use direct formulas
    // coeffs is now [1, a_{n-1}, ..., a_0] (monic)
    match degree {
        1 => {
            // Linear: x + a_0 = 0 => x = -a_0
            Some(RumpyArray::from_vec(vec![-coeffs[1]], DType::float64()))
        }
        2 => {
            // Quadratic: x^2 + bx + c = 0
            // x = (-b ± sqrt(b² - 4c)) / 2
            let b = coeffs[1];
            let c = coeffs[2];
            let disc = b * b - 4.0 * c;
            if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                let r1 = (-b + sqrt_disc) / 2.0;
                let r2 = (-b - sqrt_disc) / 2.0;
                // Sort by magnitude descending (NumPy convention)
                let mut roots = vec![r1, r2];
                roots.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));
                Some(RumpyArray::from_vec(roots, DType::float64()))
            } else {
                // Complex roots - return empty for now (we don't support complex output)
                // A more complete implementation would return complex128
                Some(RumpyArray::from_vec(Vec::new(), DType::float64()))
            }
        }
        _ => {
            // For higher degrees, use companion matrix eigenvalue approach
            roots_via_companion(&coeffs, degree)
        }
    }
}

/// Compute roots via companion matrix eigenvalues for degree >= 3.
fn roots_via_companion(coeffs: &[f64], degree: usize) -> Option<RumpyArray> {
    // Build companion matrix
    // For monic polynomial x^n + a_{n-1}*x^{n-1} + ... + a_0:
    // C = [[0, 0, ..., 0, -a_0],
    //      [1, 0, ..., 0, -a_1],
    //      [0, 1, ..., 0, -a_2],
    //      ...
    //      [0, 0, ..., 1, -a_{n-1}]]
    let mut companion_data = vec![0.0; degree * degree];

    // Sub-diagonal ones
    for i in 1..degree {
        companion_data[i * degree + (i - 1)] = 1.0;
    }

    // Last column: negative of normalized coefficients (except leading)
    for i in 0..degree {
        companion_data[i * degree + (degree - 1)] = -coeffs[degree - i];
    }

    let companion = RumpyArray::from_vec(companion_data, DType::float64())
        .reshape(vec![degree, degree])?;

    // Use QR iteration for eigenvalues
    // 50 iterations is typically sufficient for convergence
    let eigenvalues = qr_eigenvalues(&companion, 50)?;

    Some(eigenvalues)
}

/// QR algorithm for eigenvalues of a general matrix.
/// Returns real eigenvalues (may miss complex conjugate pairs).
fn qr_eigenvalues(a: &RumpyArray, max_iter: usize) -> Option<RumpyArray> {
    let n = a.shape()[0];
    if n == 0 {
        return Some(RumpyArray::from_vec(Vec::new(), DType::float64()));
    }

    let mut matrix = a.copy();

    // Plain QR iteration (no shifts)
    for _ in 0..max_iter {
        let (q, r) = linalg::qr(&matrix)?;
        // A_new = R * Q
        matrix = crate::ops::matmul::matmul(&r, &q)?;
    }

    // Extract diagonal (eigenvalues for converged matrix)
    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        eigenvalues.push(matrix.get_element(&[i, i]));
    }

    // Sort by magnitude (largest first, matching NumPy convention)
    eigenvalues.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));

    Some(RumpyArray::from_vec(eigenvalues, DType::float64()))
}
