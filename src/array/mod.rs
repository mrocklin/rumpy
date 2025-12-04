pub mod buffer;
pub mod dtype;
pub mod flags;

use std::sync::Arc;

pub use buffer::ArrayBuffer;
pub use dtype::{promote_dtype, DType, DTypeOps};
pub use flags::ArrayFlags;

/// Core N-dimensional array type.
#[derive(Clone)]
pub struct RumpyArray {
    /// Shared ownership of underlying buffer (enables views).
    buffer: Arc<ArrayBuffer>,
    /// Byte offset into buffer where this array's data starts.
    offset: usize,
    /// Shape: size in each dimension.
    shape: Vec<usize>,
    /// Strides: bytes to skip for each dimension (signed for negative strides).
    strides: Vec<isize>,
    /// Element type.
    dtype: DType,
    /// Memory layout and permission flags.
    flags: ArrayFlags,
}

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
            unsafe { ops.write_element(ptr, i, v); }
        }
        arr
    }

    /// Create array from Vec with given shape.
    pub fn from_vec_with_shape(data: Vec<f64>, shape: Vec<usize>, dtype: DType) -> Self {
        let arr = Self::from_vec(data, dtype);
        arr.reshape(shape).unwrap_or(arr)
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
        let one = ops.one_value();

        for i in 0..size {
            unsafe { ops.write_element(ptr, i, one); }
        }
    }

    /// Get array shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get array strides in bytes.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Get number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get element type.
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Get total number of elements.
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get size of one element in bytes.
    pub fn itemsize(&self) -> usize {
        self.dtype.itemsize()
    }

    /// Get total size in bytes.
    pub fn nbytes(&self) -> usize {
        self.size() * self.itemsize()
    }

    /// Get flags.
    pub fn flags(&self) -> ArrayFlags {
        self.flags
    }

    /// Get pointer to start of array data.
    pub fn data_ptr(&self) -> *const u8 {
        unsafe { self.buffer.as_ptr().add(self.offset) }
    }

    /// Get mutable reference to buffer (for internal use).
    pub(crate) fn buffer_mut(&mut self) -> &mut Arc<ArrayBuffer> {
        &mut self.buffer
    }

    /// Check if array is C-contiguous.
    pub fn is_c_contiguous(&self) -> bool {
        self.flags.contains(ArrayFlags::C_CONTIGUOUS)
    }

    /// Check if array is F-contiguous.
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.contains(ArrayFlags::F_CONTIGUOUS)
    }

    /// Create a view with new offset, shape, and strides, sharing the buffer.
    pub fn view_with(&self, offset_delta: usize, shape: Vec<usize>, strides: Vec<isize>) -> Self {
        let mut flags = ArrayFlags::WRITEABLE;
        if is_c_contiguous(&shape, &strides, self.dtype.itemsize()) {
            flags |= ArrayFlags::C_CONTIGUOUS;
        }
        if is_f_contiguous(&shape, &strides, self.dtype.itemsize()) {
            flags |= ArrayFlags::F_CONTIGUOUS;
        }

        Self {
            buffer: Arc::clone(&self.buffer),
            offset: self.offset + offset_delta,
            shape,
            strides,
            dtype: self.dtype.clone(),
            flags,
        }
    }

    /// Slice along one axis: arr[start:stop:step] for that axis.
    /// Expects pre-normalized indices from PySlice.indices().
    /// Returns a view (no copy).
    pub fn slice_axis(&self, axis: usize, start: isize, stop: isize, step: isize) -> Self {
        // Compute new length
        let new_len = if step > 0 {
            if stop > start { (stop - start + step - 1) / step } else { 0 }
        } else {
            if start > stop { (start - stop - step - 1) / (-step) } else { 0 }
        };
        let new_len = new_len.max(0) as usize;

        // Compute offset delta (bytes to skip to reach start element)
        let old_stride = self.strides[axis];
        let offset_delta = (start as usize) * (old_stride.unsigned_abs());

        // New stride = old_stride * step
        let mut new_strides = self.strides.clone();
        new_strides[axis] = old_stride * step;

        let mut new_shape = self.shape.clone();
        new_shape[axis] = new_len;

        self.view_with(offset_delta, new_shape, new_strides)
    }

    /// Reshape array. Returns a view if contiguous, otherwise would need copy.
    /// For now, only works on contiguous arrays.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Option<Self> {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        if old_size != new_size {
            return None;
        }
        if !self.is_c_contiguous() && !self.is_f_contiguous() {
            return None; // Would need copy
        }

        let new_strides = if self.is_c_contiguous() {
            compute_c_strides(&new_shape, self.dtype.itemsize())
        } else {
            compute_f_strides(&new_shape, self.dtype.itemsize())
        };

        Some(self.view_with(0, new_shape, new_strides))
    }

    /// Transpose the array (reverse axes). Returns a view.
    pub fn transpose(&self) -> Self {
        let new_shape: Vec<usize> = self.shape.iter().rev().copied().collect();
        let new_strides: Vec<isize> = self.strides.iter().rev().copied().collect();
        self.view_with(0, new_shape, new_strides)
    }

    /// Broadcast array to a new shape. Returns a view with zero strides for broadcast dims.
    /// Returns None if shape is incompatible.
    pub fn broadcast_to(&self, new_shape: &[usize]) -> Option<Self> {
        if new_shape.len() < self.ndim() {
            return None;
        }

        let mut new_strides = vec![0isize; new_shape.len()];
        let offset = new_shape.len() - self.ndim();

        // Check compatibility and compute strides
        for i in 0..self.ndim() {
            let old_dim = self.shape[i];
            let new_dim = new_shape[offset + i];

            if old_dim == new_dim {
                new_strides[offset + i] = self.strides[i];
            } else if old_dim == 1 {
                new_strides[offset + i] = 0; // Broadcast: zero stride
            } else {
                return None; // Incompatible
            }
        }

        // Leading dimensions (prepended 1s) already have zero stride from vec! initialization

        Some(self.view_with(0, new_shape.to_vec(), new_strides))
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
            unsafe { ops.write_element(ptr, i, val); }
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
            unsafe { ops.write_element(ptr, i, val); }
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
        let one = ops.one_value();

        for i in 0..n {
            let idx = i * n + i;
            unsafe { ops.write_element(ptr, idx, one); }
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
            unsafe { ops.write_element(ptr, i, value); }
        }
        arr
    }

    /// Get single element by indices. Returns as f64 for simplicity.
    pub fn get_element(&self, indices: &[usize]) -> f64 {
        assert_eq!(indices.len(), self.ndim(), "wrong number of indices");
        let mut byte_offset = self.offset as isize;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "index out of bounds");
            byte_offset += (idx as isize) * self.strides[i];
        }

        let ptr = self.buffer.as_ptr();
        unsafe { self.dtype.ops().read_element(ptr, byte_offset) }
    }
}

/// Compute C-order (row-major) strides for given shape and itemsize.
pub fn compute_c_strides(shape: &[usize], itemsize: usize) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0isize; shape.len()];
    let mut stride = itemsize as isize;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i] as isize;
    }
    strides
}

/// Compute F-order (column-major) strides for given shape and itemsize.
pub fn compute_f_strides(shape: &[usize], itemsize: usize) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0isize; shape.len()];
    let mut stride = itemsize as isize;
    for i in 0..shape.len() {
        strides[i] = stride;
        stride *= shape[i] as isize;
    }
    strides
}

/// Check if shape/strides represent C-contiguous layout.
fn is_c_contiguous(shape: &[usize], strides: &[isize], itemsize: usize) -> bool {
    if shape.is_empty() {
        return true;
    }
    let mut expected = itemsize as isize;
    for i in (0..shape.len()).rev() {
        if shape[i] == 0 {
            return true; // Empty array is contiguous
        }
        if shape[i] != 1 && strides[i] != expected {
            return false;
        }
        expected *= shape[i] as isize;
    }
    true
}

/// Check if shape/strides represent F-contiguous layout.
fn is_f_contiguous(shape: &[usize], strides: &[isize], itemsize: usize) -> bool {
    if shape.is_empty() {
        return true;
    }
    let mut expected = itemsize as isize;
    for i in 0..shape.len() {
        if shape[i] == 0 {
            return true;
        }
        if shape[i] != 1 && strides[i] != expected {
            return false;
        }
        expected *= shape[i] as isize;
    }
    true
}

/// Increment indices in row-major (C) order.
pub(crate) fn increment_indices(indices: &mut [usize], shape: &[usize]) {
    for i in (0..indices.len()).rev() {
        indices[i] += 1;
        if indices[i] < shape[i] {
            return;
        }
        indices[i] = 0;
    }
}

/// Write a value to buffer at linear index.
#[inline]
pub(crate) unsafe fn write_element(ptr: *mut u8, idx: usize, val: f64, dtype: &DType) {
    dtype.ops().write_element(ptr, idx, val);
}

/// Read element from buffer at byte offset.
#[inline]
pub(crate) unsafe fn read_element(ptr: *const u8, offset: isize, dtype: &DType) -> f64 {
    dtype.ops().read_element(ptr, offset)
}

/// Compute broadcast shape from two input shapes.
/// Returns None if shapes are incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = vec![0usize; max_ndim];

    // Align from the right
    for i in 0..max_ndim {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim == b_dim {
            result[max_ndim - 1 - i] = a_dim;
        } else if a_dim == 1 {
            result[max_ndim - 1 - i] = b_dim;
        } else if b_dim == 1 {
            result[max_ndim - 1 - i] = a_dim;
        } else {
            return None; // Incompatible
        }
    }
    Some(result)
}
