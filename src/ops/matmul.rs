//! Matrix multiplication gufunc.
//!
//! Implements batched matrix multiplication with signature "(m,n),(n,p)->(m,p)".

use crate::array::RumpyArray;
use crate::ops::gufunc::{gufunc_call, GufuncKernel, GufuncSignature};

/// Matrix multiplication kernel.
///
/// Signature: (m,n),(n,p)->(m,p)
///
/// This is a simple triple-loop implementation. For production use,
/// this could be replaced with a BLAS-backed implementation.
pub struct MatmulKernel {
    sig: GufuncSignature,
}

impl MatmulKernel {
    pub fn new() -> Self {
        MatmulKernel {
            sig: GufuncSignature::parse("(m,n),(n,p)->(m,p)").unwrap(),
        }
    }
}

impl Default for MatmulKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GufuncKernel for MatmulKernel {
    fn signature(&self) -> &GufuncSignature {
        &self.sig
    }

    fn call(&self, inputs: &[RumpyArray], outputs: &mut [RumpyArray]) {
        let a = &inputs[0];
        let b = &inputs[1];
        let c = &mut outputs[0];

        let m = a.shape()[0];
        let n = a.shape()[1];
        let p = b.shape()[1];

        // Get output data pointer and strides for writing
        let c_ptr = c.data_ptr() as *mut u8;
        let c_stride_0 = c.strides()[0];
        let c_stride_1 = c.strides()[1];

        // Simple triple-loop matmul: C[i,j] = sum_k A[i,k] * B[k,j]
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a.get_element(&[i, k]) * b.get_element(&[k, j]);
                }

                // Write to output using strides
                let byte_offset = (i as isize) * c_stride_0 + (j as isize) * c_stride_1;
                unsafe {
                    let dst = c_ptr.offset(byte_offset);
                    // Assume float64 for now - write directly
                    *(dst as *mut f64) = sum;
                }
            }
        }
    }
}

/// Matrix multiplication with broadcasting.
///
/// Supports batched matmul: [B, M, N] @ [B, N, P] -> [B, M, P]
/// Also supports broadcasting: [M, N] @ [B, N, P] -> [B, M, P]
///
/// # Examples
///
/// 2D × 2D:
/// ```ignore
/// let a = rp.array([[1, 2], [3, 4]]);  // (2, 2)
/// let b = rp.array([[5, 6], [7, 8]]);  // (2, 2)
/// let c = rp.matmul(a, b);             // (2, 2)
/// ```
///
/// Batched:
/// ```ignore
/// let a = rp.zeros([3, 2, 4]);  // (3, 2, 4)
/// let b = rp.zeros([3, 4, 5]);  // (3, 4, 5)
/// let c = rp.matmul(a, b);      // (3, 2, 5)
/// ```
pub fn matmul(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    // Handle 1D inputs by expanding dims
    let (a_expanded, b_expanded, squeeze_result) = expand_for_matmul(a, b)?;

    let kernel = MatmulKernel::new();
    let mut results = gufunc_call(&kernel, &[&a_expanded, &b_expanded])?;

    let result = results.pop()?;

    // Squeeze result if we expanded inputs
    if squeeze_result {
        Some(result.squeeze())
    } else {
        Some(result)
    }
}

/// Expand 1D arrays for matmul and return whether to squeeze result.
///
/// NumPy semantics:
/// - 1D @ 1D -> scalar (inner product)
/// - 1D @ 2D -> 1D (prepend 1 to first, squeeze first dim of result)
/// - 2D @ 1D -> 1D (append 1 to second, squeeze last dim of result)
fn expand_for_matmul(
    a: &RumpyArray,
    b: &RumpyArray,
) -> Option<(RumpyArray, RumpyArray, bool)> {
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    match (a_ndim, b_ndim) {
        (1, 1) => {
            // Inner product: (n,) @ (n,) -> ()
            // Expand to (1, n) @ (n, 1) -> (1, 1), then squeeze
            let a_exp = a.expand_dims(0)?;
            let b_exp = b.expand_dims(1)?;
            Some((a_exp, b_exp, true))
        }
        (1, _) => {
            // (n,) @ (..., n, p) -> (..., p)
            // Expand to (1, n) @ (..., n, p) -> (..., 1, p), squeeze axis -2
            let a_exp = a.expand_dims(0)?;
            Some((a_exp, b.clone(), true))
        }
        (_, 1) => {
            // (..., m, n) @ (n,) -> (..., m)
            // Expand to (..., m, n) @ (n, 1) -> (..., m, 1), squeeze last
            let b_exp = b.expand_dims(1)?;
            Some((a.clone(), b_exp, true))
        }
        _ => {
            // Standard 2D+ case
            Some((a.clone(), b.clone(), false))
        }
    }
}
