//! Matrix multiplication gufunc.
//!
//! Implements batched matrix multiplication with signature "(m,n),(n,p)->(m,p)".
//! Uses faer for optimized matrix multiplication.

use crate::array::RumpyArray;
use crate::ops::gufunc::{gufunc_call, GufuncKernel, GufuncSignature};
use faer::mat;

/// Matrix multiplication kernel using faer.
///
/// Signature: (m,n),(n,p)->(m,p)
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

impl GufuncKernel for MatmulKernel {
    fn signature(&self) -> &GufuncSignature {
        &self.sig
    }

    fn call(&self, inputs: &[RumpyArray], outputs: &mut [RumpyArray]) {
        let a = &inputs[0];
        let b = &inputs[1];
        let c = &outputs[0];

        let m = a.shape()[0];
        let n = a.shape()[1];
        let p = b.shape()[1];

        // Convert byte strides to element strides
        let elem_size = std::mem::size_of::<f64>() as isize;
        let a_row_stride = a.strides()[0] / elem_size;
        let a_col_stride = a.strides()[1] / elem_size;
        let b_row_stride = b.strides()[0] / elem_size;
        let b_col_stride = b.strides()[1] / elem_size;
        let c_row_stride = c.strides()[0] / elem_size;
        let c_col_stride = c.strides()[1] / elem_size;

        unsafe {
            // Create faer views directly from our memory
            let fa = mat::from_raw_parts::<f64, usize, usize>(
                a.data_ptr() as *const f64,
                m, n,
                a_row_stride, a_col_stride,
            );
            let fb = mat::from_raw_parts::<f64, usize, usize>(
                b.data_ptr() as *const f64,
                n, p,
                b_row_stride, b_col_stride,
            );
            let mut fc = mat::from_raw_parts_mut::<f64, usize, usize>(
                c.data_ptr() as *mut f64,
                m, p,
                c_row_stride, c_col_stride,
            );

            // Multiply using faer: C = A * B
            faer::linalg::matmul::matmul(
                fc.as_mut(),
                fa.as_ref(),
                fb.as_ref(),
                None,  // No existing C to add
                1.0,   // alpha = 1
                faer::Parallelism::None,
            );
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
