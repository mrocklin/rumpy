//! Inner product gufunc.
//!
//! Implements inner product with signature "(n),(n)->()".

use crate::array::RumpyArray;
use crate::ops::gufunc::{gufunc_call, GufuncKernel, GufuncSignature};

/// Inner product kernel.
///
/// Signature: (n),(n)->()
pub struct InnerKernel {
    sig: GufuncSignature,
}

impl InnerKernel {
    pub fn new() -> Self {
        InnerKernel {
            sig: GufuncSignature::parse("(n),(n)->()").unwrap(),
        }
    }
}

impl GufuncKernel for InnerKernel {
    fn signature(&self) -> &GufuncSignature {
        &self.sig
    }

    fn call(&self, inputs: &[RumpyArray], outputs: &mut [RumpyArray]) {
        let a = &inputs[0];
        let b = &inputs[1];
        let c = &outputs[0];

        let n = a.shape()[0];

        // Inner product: sum_i a[i] * b[i]
        let mut sum = 0.0;
        for i in 0..n {
            sum += a.get_element(&[i]) * b.get_element(&[i]);
        }

        // Write scalar result
        let c_ptr = c.data_ptr() as *mut f64;
        unsafe {
            *c_ptr = sum;
        }
    }
}

/// Inner product of two arrays.
///
/// For 1D arrays, this is the dot product (sum of element-wise products).
/// For higher-D arrays, it's a sum over the last axis of both arrays.
///
/// # Examples
///
/// 1D inner product:
/// ```ignore
/// let a = rp.array([1, 2, 3]);
/// let b = rp.array([4, 5, 6]);
/// let c = rp.inner(a, b);  // 1*4 + 2*5 + 3*6 = 32
/// ```
///
/// Broadcasting with loop dimensions:
/// ```ignore
/// let a = rp.zeros([3, 4]);  // (3, 4) - 3 vectors of length 4
/// let b = rp.zeros([4]);     // (4,) - one vector
/// let c = rp.inner(a, b);    // (3,) - 3 inner products
/// ```
pub fn inner(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    let kernel = InnerKernel::new();
    let mut results = gufunc_call(&kernel, &[a, b])?;
    results.pop()
}
