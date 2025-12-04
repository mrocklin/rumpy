//! Outer product gufunc.
//!
//! Implements outer product with signature "(m),(n)->(m,n)".

use crate::array::RumpyArray;
use crate::ops::gufunc::{gufunc_call, GufuncKernel, GufuncSignature};

/// Outer product kernel.
///
/// Signature: (m),(n)->(m,n)
pub struct OuterKernel {
    sig: GufuncSignature,
}

impl OuterKernel {
    pub fn new() -> Self {
        OuterKernel {
            sig: GufuncSignature::parse("(m),(n)->(m,n)").unwrap(),
        }
    }
}

impl GufuncKernel for OuterKernel {
    fn signature(&self) -> &GufuncSignature {
        &self.sig
    }

    fn call(&self, inputs: &[RumpyArray], outputs: &mut [RumpyArray]) {
        let a = &inputs[0];
        let b = &inputs[1];
        let c = &outputs[0];

        let m = a.shape()[0];
        let n = b.shape()[0];

        let c_ptr = c.data_ptr() as *mut u8;
        let c_stride_0 = c.strides()[0];
        let c_stride_1 = c.strides()[1];

        // Outer product: c[i,j] = a[i] * b[j]
        for i in 0..m {
            let ai = a.get_element(&[i]);
            for j in 0..n {
                let bj = b.get_element(&[j]);
                let byte_offset = (i as isize) * c_stride_0 + (j as isize) * c_stride_1;
                unsafe {
                    let dst = c_ptr.offset(byte_offset);
                    *(dst as *mut f64) = ai * bj;
                }
            }
        }
    }
}

/// Outer product of two vectors.
///
/// Computes the outer product of two 1D arrays, producing a 2D matrix.
/// For higher-D arrays, operates on the last axis (flattens in NumPy,
/// but here we require 1D inputs).
///
/// # Examples
///
/// ```ignore
/// let a = rp.array([1, 2, 3]);
/// let b = rp.array([4, 5]);
/// let c = rp.outer(a, b);
/// // [[4, 5],
/// //  [8, 10],
/// //  [12, 15]]
/// ```
pub fn outer(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    // NumPy flattens inputs; we follow that behavior
    let a_flat = a.reshape(vec![a.size()])?;
    let b_flat = b.reshape(vec![b.size()])?;

    let kernel = OuterKernel::new();
    let mut results = gufunc_call(&kernel, &[&a_flat, &b_flat])?;
    results.pop()
}
