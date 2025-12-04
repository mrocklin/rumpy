pub mod buffer;
pub mod dtype;
pub mod flags;

use std::sync::Arc;

pub use buffer::ArrayBuffer;
pub use dtype::DType;
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

        unsafe {
            match self.dtype {
                DType::Float32 => {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut f32, size);
                    slice.fill(1.0);
                }
                DType::Float64 => {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut f64, size);
                    slice.fill(1.0);
                }
                DType::Int32 => {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut i32, size);
                    slice.fill(1);
                }
                DType::Int64 => {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut i64, size);
                    slice.fill(1);
                }
                DType::Bool => {
                    let slice = std::slice::from_raw_parts_mut(ptr, size);
                    slice.fill(1); // true = 1
                }
            }
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
        self.dtype
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
            dtype: self.dtype,
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

        let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
        let ptr = buffer.as_mut_ptr();

        unsafe {
            match dtype {
                DType::Float64 => {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut f64, n);
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = start + (i as f64) * step;
                    }
                }
                DType::Float32 => {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut f32, n);
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = (start + (i as f64) * step) as f32;
                    }
                }
                DType::Int64 => {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut i64, n);
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = (start + (i as f64) * step) as i64;
                    }
                }
                DType::Int32 => {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut i32, n);
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = (start + (i as f64) * step) as i32;
                    }
                }
                DType::Bool => {
                    let slice = std::slice::from_raw_parts_mut(ptr, n);
                    for (i, v) in slice.iter_mut().enumerate() {
                        *v = ((start + (i as f64) * step) != 0.0) as u8;
                    }
                }
            }
        }
        arr
    }

    /// Get single element by indices. Returns as f64 for simplicity.
    pub fn get_element(&self, indices: &[usize]) -> f64 {
        assert_eq!(indices.len(), self.ndim(), "wrong number of indices");
        let mut byte_offset = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "index out of bounds");
            byte_offset = (byte_offset as isize + (idx as isize) * self.strides[i]) as usize;
        }

        let ptr = self.buffer.as_ptr();
        unsafe {
            match self.dtype {
                DType::Float64 => *ptr.add(byte_offset).cast::<f64>(),
                DType::Float32 => *ptr.add(byte_offset).cast::<f32>() as f64,
                DType::Int64 => *ptr.add(byte_offset).cast::<i64>() as f64,
                DType::Int32 => *ptr.add(byte_offset).cast::<i32>() as f64,
                DType::Bool => *ptr.add(byte_offset) as f64,
            }
        }
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
