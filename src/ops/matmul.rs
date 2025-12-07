//! Matrix multiplication gufunc.
//!
//! Implements batched matrix multiplication with signature "(m,n),(n,p)->(m,p)".
//! Uses BLAS (Accelerate on macOS) when available, falls back to faer.

use crate::array::{DType, RumpyArray};
use crate::ops::gufunc::{gufunc_call, GufuncKernel, GufuncSignature};

// Link BLAS when accelerate feature is enabled
#[cfg(feature = "accelerate")]
extern crate blas_src;

/// Matrix multiplication kernel.
///
/// Signature: (m,n),(n,p)->(m,p)
/// Uses BLAS (sgemm/dgemm/cgemm/zgemm) when accelerate feature is enabled, otherwise faer.
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
        let dtype = c.dtype();

        #[cfg(feature = "accelerate")]
        {
            // Use BLAS for C-contiguous arrays
            if a.is_c_contiguous() && b.is_c_contiguous() && c.is_c_contiguous() {
                // BLAS is column-major, we're row-major.
                // C = A * B in row-major  <==>  C^T = B^T * A^T in column-major
                // Since C^T with row-major strides looks like C with col-major strides,
                // we compute: C = B * A with swapped dimensions

                if dtype == DType::float32() {
                    unsafe {
                        blas::sgemm(
                            b'N', b'N',
                            p as i32, m as i32, n as i32,
                            1.0f32,
                            std::slice::from_raw_parts(b.data_ptr() as *const f32, n * p),
                            p as i32,
                            std::slice::from_raw_parts(a.data_ptr() as *const f32, m * n),
                            n as i32,
                            0.0f32,
                            std::slice::from_raw_parts_mut(c.data_ptr() as *mut f32, m * p),
                            p as i32,
                        );
                    }
                    return;
                }

                if dtype == DType::float64() {
                    unsafe {
                        blas::dgemm(
                            b'N', b'N',
                            p as i32, m as i32, n as i32,
                            1.0f64,
                            std::slice::from_raw_parts(b.data_ptr() as *const f64, n * p),
                            p as i32,
                            std::slice::from_raw_parts(a.data_ptr() as *const f64, m * n),
                            n as i32,
                            0.0f64,
                            std::slice::from_raw_parts_mut(c.data_ptr() as *mut f64, m * p),
                            p as i32,
                        );
                    }
                    return;
                }

                if dtype == DType::complex64() {
                    unsafe {
                        blas::cgemm(
                            b'N', b'N',
                            p as i32, m as i32, n as i32,
                            blas::c32::new(1.0, 0.0),
                            std::slice::from_raw_parts(b.data_ptr() as *const blas::c32, n * p),
                            p as i32,
                            std::slice::from_raw_parts(a.data_ptr() as *const blas::c32, m * n),
                            n as i32,
                            blas::c32::new(0.0, 0.0),
                            std::slice::from_raw_parts_mut(c.data_ptr() as *mut blas::c32, m * p),
                            p as i32,
                        );
                    }
                    return;
                }

                if dtype == DType::complex128() {
                    unsafe {
                        blas::zgemm(
                            b'N', b'N',
                            p as i32, m as i32, n as i32,
                            blas::c64::new(1.0, 0.0),
                            std::slice::from_raw_parts(b.data_ptr() as *const blas::c64, n * p),
                            p as i32,
                            std::slice::from_raw_parts(a.data_ptr() as *const blas::c64, m * n),
                            n as i32,
                            blas::c64::new(0.0, 0.0),
                            std::slice::from_raw_parts_mut(c.data_ptr() as *mut blas::c64, m * p),
                            p as i32,
                        );
                    }
                    return;
                }
            }
        }

        // Fallback to faer for float/complex, generic loop for integers
        if dtype == DType::float64() {
            matmul_faer_f64(a, b, c, m, n, p);
        } else if dtype == DType::float32() {
            matmul_faer_f32(a, b, c, m, n, p);
        } else if dtype == DType::complex64() {
            matmul_faer_c32(a, b, c, m, n, p);
        } else if dtype == DType::complex128() {
            matmul_faer_c64(a, b, c, m, n, p);
        } else if dtype == DType::int64() {
            matmul_generic::<i64>(a, b, c, m, n, p);
        } else if dtype == DType::int32() {
            matmul_generic::<i32>(a, b, c, m, n, p);
        } else if dtype == DType::bool() {
            matmul_bool(a, b, c, m, n, p);
        } else {
            // Unsupported dtype - zero output
        }
    }
}

/// Bool matmul: C[i,j] = any(A[i,k] AND B[k,j] for all k)
fn matmul_bool(a: &RumpyArray, b: &RumpyArray, c: &RumpyArray, m: usize, n: usize, p: usize) {
    let a_row_stride = a.strides()[0] as usize;
    let a_col_stride = a.strides()[1] as usize;
    let b_row_stride = b.strides()[0] as usize;
    let b_col_stride = b.strides()[1] as usize;
    let c_row_stride = c.strides()[0] as usize;
    let c_col_stride = c.strides()[1] as usize;

    let a_ptr = a.data_ptr();
    let b_ptr = b.data_ptr();
    let c_ptr = c.data_ptr() as *mut u8;

    for i in 0..m {
        for j in 0..p {
            let mut result = false;
            for k in 0..n {
                unsafe {
                    let a_val = *a_ptr.add(i * a_row_stride + k * a_col_stride) != 0;
                    let b_val = *b_ptr.add(k * b_row_stride + j * b_col_stride) != 0;
                    if a_val && b_val {
                        result = true;
                        break; // Short-circuit: any true is enough
                    }
                }
            }
            unsafe {
                *c_ptr.add(i * c_row_stride + j * c_col_stride) = result as u8;
            }
        }
    }
}

/// Generic matmul for integer types (no BLAS/faer support)
fn matmul_generic<T>(a: &RumpyArray, b: &RumpyArray, c: &RumpyArray, m: usize, n: usize, p: usize)
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let elem_size = std::mem::size_of::<T>() as isize;
    let a_row_stride = (a.strides()[0] / elem_size) as usize;
    let a_col_stride = (a.strides()[1] / elem_size) as usize;
    let b_row_stride = (b.strides()[0] / elem_size) as usize;
    let b_col_stride = (b.strides()[1] / elem_size) as usize;
    let c_row_stride = (c.strides()[0] / elem_size) as usize;
    let c_col_stride = (c.strides()[1] / elem_size) as usize;

    let a_ptr = a.data_ptr() as *const T;
    let b_ptr = b.data_ptr() as *const T;
    let c_ptr = c.data_ptr() as *mut T;

    for i in 0..m {
        for j in 0..p {
            let mut sum = T::default();
            for k in 0..n {
                unsafe {
                    let a_val = *a_ptr.add(i * a_row_stride + k * a_col_stride);
                    let b_val = *b_ptr.add(k * b_row_stride + j * b_col_stride);
                    sum = sum + a_val * b_val;
                }
            }
            unsafe {
                *c_ptr.add(i * c_row_stride + j * c_col_stride) = sum;
            }
        }
    }
}

/// Fallback matmul using faer for f64
fn matmul_faer_f64(a: &RumpyArray, b: &RumpyArray, c: &RumpyArray, m: usize, n: usize, p: usize) {
    use faer::mat;

    let elem_size = std::mem::size_of::<f64>() as isize;
    let a_row_stride = a.strides()[0] / elem_size;
    let a_col_stride = a.strides()[1] / elem_size;
    let b_row_stride = b.strides()[0] / elem_size;
    let b_col_stride = b.strides()[1] / elem_size;
    let c_row_stride = c.strides()[0] / elem_size;
    let c_col_stride = c.strides()[1] / elem_size;

    unsafe {
        let fa = mat::from_raw_parts::<f64, usize, usize>(
            a.data_ptr() as *const f64, m, n, a_row_stride, a_col_stride,
        );
        let fb = mat::from_raw_parts::<f64, usize, usize>(
            b.data_ptr() as *const f64, n, p, b_row_stride, b_col_stride,
        );
        let mut fc = mat::from_raw_parts_mut::<f64, usize, usize>(
            c.data_ptr() as *mut f64, m, p, c_row_stride, c_col_stride,
        );

        faer::linalg::matmul::matmul(
            fc.as_mut(), fa.as_ref(), fb.as_ref(),
            None, 1.0, faer::Parallelism::Rayon(0),
        );
    }
}

/// Fallback matmul using faer for f32
fn matmul_faer_f32(a: &RumpyArray, b: &RumpyArray, c: &RumpyArray, m: usize, n: usize, p: usize) {
    use faer::mat;

    let elem_size = std::mem::size_of::<f32>() as isize;
    let a_row_stride = a.strides()[0] / elem_size;
    let a_col_stride = a.strides()[1] / elem_size;
    let b_row_stride = b.strides()[0] / elem_size;
    let b_col_stride = b.strides()[1] / elem_size;
    let c_row_stride = c.strides()[0] / elem_size;
    let c_col_stride = c.strides()[1] / elem_size;

    unsafe {
        let fa = mat::from_raw_parts::<f32, usize, usize>(
            a.data_ptr() as *const f32, m, n, a_row_stride, a_col_stride,
        );
        let fb = mat::from_raw_parts::<f32, usize, usize>(
            b.data_ptr() as *const f32, n, p, b_row_stride, b_col_stride,
        );
        let mut fc = mat::from_raw_parts_mut::<f32, usize, usize>(
            c.data_ptr() as *mut f32, m, p, c_row_stride, c_col_stride,
        );

        faer::linalg::matmul::matmul(
            fc.as_mut(), fa.as_ref(), fb.as_ref(),
            None, 1.0f32, faer::Parallelism::Rayon(0),
        );
    }
}

/// Fallback matmul using faer for complex64
fn matmul_faer_c32(a: &RumpyArray, b: &RumpyArray, c: &RumpyArray, m: usize, n: usize, p: usize) {
    use faer::mat;
    use faer::complex_native::c32;

    let elem_size = std::mem::size_of::<c32>() as isize;
    let a_row_stride = a.strides()[0] / elem_size;
    let a_col_stride = a.strides()[1] / elem_size;
    let b_row_stride = b.strides()[0] / elem_size;
    let b_col_stride = b.strides()[1] / elem_size;
    let c_row_stride = c.strides()[0] / elem_size;
    let c_col_stride = c.strides()[1] / elem_size;

    unsafe {
        let fa = mat::from_raw_parts::<c32, usize, usize>(
            a.data_ptr() as *const c32, m, n, a_row_stride, a_col_stride,
        );
        let fb = mat::from_raw_parts::<c32, usize, usize>(
            b.data_ptr() as *const c32, n, p, b_row_stride, b_col_stride,
        );
        let mut fc = mat::from_raw_parts_mut::<c32, usize, usize>(
            c.data_ptr() as *mut c32, m, p, c_row_stride, c_col_stride,
        );

        faer::linalg::matmul::matmul(
            fc.as_mut(), fa.as_ref(), fb.as_ref(),
            None, c32::new(1.0, 0.0), faer::Parallelism::Rayon(0),
        );
    }
}

/// Fallback matmul using faer for complex128
fn matmul_faer_c64(a: &RumpyArray, b: &RumpyArray, c: &RumpyArray, m: usize, n: usize, p: usize) {
    use faer::mat;
    use faer::complex_native::c64;

    let elem_size = std::mem::size_of::<c64>() as isize;
    let a_row_stride = a.strides()[0] / elem_size;
    let a_col_stride = a.strides()[1] / elem_size;
    let b_row_stride = b.strides()[0] / elem_size;
    let b_col_stride = b.strides()[1] / elem_size;
    let c_row_stride = c.strides()[0] / elem_size;
    let c_col_stride = c.strides()[1] / elem_size;

    unsafe {
        let fa = mat::from_raw_parts::<c64, usize, usize>(
            a.data_ptr() as *const c64, m, n, a_row_stride, a_col_stride,
        );
        let fb = mat::from_raw_parts::<c64, usize, usize>(
            b.data_ptr() as *const c64, n, p, b_row_stride, b_col_stride,
        );
        let mut fc = mat::from_raw_parts_mut::<c64, usize, usize>(
            c.data_ptr() as *mut c64, m, p, c_row_stride, c_col_stride,
        );

        faer::linalg::matmul::matmul(
            fc.as_mut(), fa.as_ref(), fb.as_ref(),
            None, c64::new(1.0, 0.0), faer::Parallelism::Rayon(0),
        );
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
