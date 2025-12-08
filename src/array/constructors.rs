//! Array constructor methods: zeros, ones, full, arange, linspace, eye, from_vec.

use std::sync::Arc;

use super::buffer::ArrayBuffer;
use super::dtype::DType;
use super::flags::ArrayFlags;
use super::{compute_c_strides, RumpyArray};

impl RumpyArray {
    /// Create a new array filled with zeros.
    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let size: usize = shape.iter().product();
        let nbytes = size * dtype.itemsize();
        let buffer = Arc::new(ArrayBuffer::new(nbytes));
        let strides = compute_c_strides(&shape, dtype.itemsize());

        Self {
            buffer,
            offset: 0,
            shape,
            strides,
            dtype,
            flags: ArrayFlags::default(),
        }
    }

    /// Create array from Vec of f64 values (1D).
    pub fn from_vec(data: Vec<f64>, dtype: DType) -> Self {
        let n = data.len();
        let mut arr = Self::zeros(vec![n], dtype);
        if n == 0 {
            return arr;
        }

        let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
        let ptr = buffer.as_mut_ptr();
        let ops = arr.dtype.ops();

        for (i, &v) in data.iter().enumerate() {
            unsafe { ops.write_f64(ptr, i, v); }
        }
        arr
    }

    /// Create array from Vec with given shape.
    pub fn from_vec_with_shape(data: Vec<f64>, shape: Vec<usize>, dtype: DType) -> Self {
        let arr = Self::from_vec(data, dtype);
        arr.reshape(shape).unwrap_or(arr)
    }

    /// Create array from Vec of i64 values (1D) with given dtype.
    /// Uses write_i64 to preserve precision for integer types.
    pub fn from_vec_i64(data: Vec<i64>, dtype: DType) -> Self {
        let n = data.len();
        let mut arr = Self::zeros(vec![n], dtype);
        if n == 0 {
            return arr;
        }

        let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
        let ptr = buffer.as_mut_ptr();
        let ops = arr.dtype.ops();

        for (i, &v) in data.iter().enumerate() {
            unsafe { ops.write_i64(ptr, i, v); }
        }
        arr
    }

    /// Create array from Vec of i64 with given shape.
    pub fn from_vec_i64_with_shape(data: Vec<i64>, shape: Vec<usize>, dtype: DType) -> Self {
        let arr = Self::from_vec_i64(data, dtype);
        arr.reshape(shape).unwrap_or(arr)
    }

    /// Create array from slice of i64 values (1D, as Int64 dtype).
    pub fn from_slice_i64(data: &[i64]) -> Self {
        let n = data.len();
        let mut arr = Self::zeros(vec![n], DType::int64());
        if n == 0 {
            return arr;
        }

        let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
        let ptr = buffer.as_mut_ptr();
        let ops = arr.dtype.ops();

        for (i, &v) in data.iter().enumerate() {
            unsafe { ops.write_i64(ptr, i, v); }
        }
        arr
    }

    /// Create a new array filled with ones.
    pub fn ones(shape: Vec<usize>, dtype: DType) -> Self {
        let mut arr = Self::zeros(shape, dtype);
        arr.fill_ones();
        arr
    }

    /// Fill array with ones (dtype-aware).
    fn fill_ones(&mut self) {
        let buffer = Arc::get_mut(&mut self.buffer).expect("buffer must be unique for fill");
        let ptr = buffer.as_mut_ptr();
        let size: usize = self.shape.iter().product();
        let ops = self.dtype.ops();

        for i in 0..size {
            unsafe { ops.write_one(ptr, i); }
        }
    }

    /// Create array with evenly spaced values [start, start+step, start+2*step, ...).
    pub fn arange(start: f64, stop: f64, step: f64, dtype: DType) -> Self {
        let n = ((stop - start) / step).ceil().max(0.0) as usize;
        let mut arr = Self::zeros(vec![n], dtype);

        if n == 0 {
            return arr;
        }

        let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
        let ptr = buffer.as_mut_ptr();
        let ops = arr.dtype.ops();

        for i in 0..n {
            let val = start + (i as f64) * step;
            unsafe { ops.write_f64(ptr, i, val); }
        }
        arr
    }

    /// Create array with evenly spaced values over interval [start, stop].
    pub fn linspace(start: f64, stop: f64, num: usize, dtype: DType) -> Self {
        let mut arr = Self::zeros(vec![num], dtype);

        if num == 0 {
            return arr;
        }

        let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
        let ptr = buffer.as_mut_ptr();
        let ops = arr.dtype.ops();

        // step = (stop - start) / (num - 1) for num > 1, else 0
        let step = if num > 1 { (stop - start) / (num - 1) as f64 } else { 0.0 };

        for i in 0..num {
            let val = start + (i as f64) * step;
            unsafe { ops.write_f64(ptr, i, val); }
        }
        arr
    }

    /// Create an identity matrix of size n x n.
    pub fn eye(n: usize, dtype: DType) -> Self {
        let mut arr = Self::zeros(vec![n, n], dtype);

        if n == 0 {
            return arr;
        }

        let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
        let ptr = buffer.as_mut_ptr();
        let ops = arr.dtype.ops();

        for i in 0..n {
            let idx = i * n + i;
            unsafe { ops.write_one(ptr, idx); }
        }
        arr
    }

    /// Create array filled with given value.
    pub fn full(shape: Vec<usize>, value: f64, dtype: DType) -> Self {
        let mut arr = Self::zeros(shape, dtype);
        let size = arr.size();

        if size == 0 {
            return arr;
        }

        let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
        let ptr = buffer.as_mut_ptr();
        let ops = arr.dtype.ops();

        for i in 0..size {
            unsafe { ops.write_f64(ptr, i, value); }
        }
        arr
    }

    /// Create array filled with given complex value.
    pub fn full_complex(shape: Vec<usize>, real: f64, imag: f64, dtype: DType) -> Self {
        let mut arr = Self::zeros(shape, dtype);
        let size = arr.size();

        if size == 0 {
            return arr;
        }

        let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
        let ptr = buffer.as_mut_ptr();
        let ops = arr.dtype.ops();

        for i in 0..size {
            unsafe { ops.write_complex(ptr, i, real, imag); }
        }
        arr
    }
}
