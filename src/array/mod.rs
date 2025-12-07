pub mod buffer;
pub mod constructors;
pub mod dtype;
pub mod flags;
pub mod format;
pub mod iter;
pub mod manipulation;
pub mod views;

use std::sync::Arc;

pub use buffer::ArrayBuffer;
pub use dtype::{promote_dtype, DType, DTypeOps};
pub use flags::ArrayFlags;
pub use iter::{AxisOffsetIter, StridedIter};
pub use manipulation::{
    append, array_split, bincount, broadcast_shapes, concatenate, convolve, delete, diagflat,
    geomspace, indices, insert, logspace, meshgrid, pad, percentile, repeat, roll, rot90, split,
    stack, tile, tri, tril, triu,
};

/// Core N-dimensional array type.
#[derive(Clone)]
pub struct RumpyArray {
    /// Shared ownership of underlying buffer (enables views).
    pub(crate) buffer: Arc<ArrayBuffer>,
    /// Byte offset into buffer where this array's data starts.
    pub(crate) offset: usize,
    /// Shape: size in each dimension.
    pub(crate) shape: Vec<usize>,
    /// Strides: bytes to skip for each dimension (signed for negative strides).
    pub(crate) strides: Vec<isize>,
    /// Element type.
    pub(crate) dtype: DType,
    /// Memory layout and permission flags.
    pub(crate) flags: ArrayFlags,
}

impl RumpyArray {
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

    /// Get mutable pointer to start of array data (for in-place operations).
    /// Safety: caller must ensure exclusive access.
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        unsafe { (self.buffer.as_ptr() as *mut u8).add(self.offset) }
    }

    /// Get mutable reference to buffer (for internal use).
    pub(crate) fn buffer_mut(&mut self) -> &mut Arc<ArrayBuffer> {
        &mut self.buffer
    }

    /// Check if this array's buffer can be reused for in-place output.
    ///
    /// Returns true if:
    /// - Shape matches the target shape
    /// - Dtype matches the target dtype
    /// - Array is C-contiguous
    /// - Offset is 0 (not a view into middle of buffer)
    /// - Arc has strong_count == 1 (no other Rust references)
    pub fn can_reuse_for_output(&self, shape: &[usize], dtype: &DType) -> bool {
        self.shape.as_slice() == shape
            && &self.dtype == dtype
            && self.is_c_contiguous()
            && self.offset == 0
            && Arc::strong_count(&self.buffer) == 1
    }

    /// Check if array is C-contiguous.
    pub fn is_c_contiguous(&self) -> bool {
        self.flags.contains(ArrayFlags::C_CONTIGUOUS)
    }

    /// Check if array is F-contiguous.
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.contains(ArrayFlags::F_CONTIGUOUS)
    }

    /// Iterate over byte offsets for each element in row-major order.
    /// More efficient than `increment_indices` for strided arrays.
    pub fn iter_offsets(&self) -> StridedIter {
        StridedIter::new(&self.shape, &self.strides, 0)
    }

    /// Iterate over base byte offsets for axis reduction.
    ///
    /// Each yielded offset is the starting point for a slice along `axis`.
    /// Use with `strides()[axis]` to iterate along the axis from each base.
    pub fn axis_offsets(&self, axis: usize) -> AxisOffsetIter {
        AxisOffsetIter::new(&self.shape, &self.strides, axis, 0)
    }

    /// Get single element by indices. Returns as f64 for simplicity.
    /// Returns 0.0 for dtypes that don't support f64 conversion (like complex).
    pub fn get_element(&self, indices: &[usize]) -> f64 {
        assert_eq!(indices.len(), self.ndim(), "wrong number of indices");
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "index out of bounds");
        }
        let byte_offset = self.byte_offset_for(indices);
        let ptr = self.data_ptr();
        unsafe { self.dtype.ops().read_f64(ptr, byte_offset).unwrap_or(0.0) }
    }

    /// Get single element by indices as complex (real, imag).
    /// For non-complex types, imag is 0.0.
    pub fn get_complex_element(&self, indices: &[usize]) -> (f64, f64) {
        assert_eq!(indices.len(), self.ndim(), "wrong number of indices");
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "index out of bounds");
        }
        let byte_offset = self.byte_offset_for(indices);
        let ptr = self.data_ptr();
        unsafe { self.dtype.ops().read_complex(ptr, byte_offset).unwrap_or((0.0, 0.0)) }
    }

    /// Set single element by indices.
    pub fn set_element(&mut self, indices: &[usize], value: f64) {
        assert_eq!(indices.len(), self.ndim(), "wrong number of indices");
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "index out of bounds");
        }
        let byte_offset = self.byte_offset_for(indices);
        let ptr = self.data_ptr() as *mut u8;
        unsafe {
            // Write directly using byte offset (handles strided arrays)
            self.dtype.ops().write_f64_at_byte_offset(ptr, byte_offset, value);
        }
    }

    /// Compute byte offset for given indices (relative to data_ptr, not buffer start).
    pub fn byte_offset_for(&self, indices: &[usize]) -> isize {
        let mut byte_offset: isize = 0;
        for (i, &idx) in indices.iter().enumerate() {
            byte_offset += (idx as isize) * self.strides[i];
        }
        byte_offset
    }

    /// Select elements where mask is true. Returns a 1D array.
    /// Mask must have same shape as self (broadcasting not supported for boolean indexing).
    pub fn select_by_mask(&self, mask: &RumpyArray) -> Option<Self> {
        use dtype::DTypeKind;

        // Mask must be bool dtype
        if mask.dtype().kind() != DTypeKind::Bool {
            return None;
        }

        // Shapes must match exactly
        if self.shape() != mask.shape() {
            return None;
        }

        let size = self.size();
        if size == 0 {
            return Some(Self::zeros(vec![0], self.dtype()));
        }

        // First pass: count true elements
        let mut count = 0usize;
        let mut mask_indices = vec![0usize; mask.ndim()];
        let mask_ptr = mask.data_ptr();
        let mask_ops = mask.dtype.ops();
        for _ in 0..size {
            let offset = mask.byte_offset_for(&mask_indices);
            if unsafe { mask_ops.is_truthy(mask_ptr, offset) } {
                count += 1;
            }
            increment_indices(&mut mask_indices, mask.shape());
        }

        // Create result array
        let mut result = Self::zeros(vec![count], self.dtype());
        if count == 0 {
            return Some(result);
        }

        // Second pass: copy selected elements
        let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
        let result_ptr = buffer.as_mut_ptr();
        let ops = result.dtype.ops();
        let src_ptr = self.data_ptr();

        let mut src_indices = vec![0usize; self.ndim()];
        let mut dst_idx = 0usize;
        for _ in 0..size {
            let mask_offset = mask.byte_offset_for(&src_indices);
            if unsafe { mask_ops.is_truthy(mask_ptr, mask_offset) } {
                let src_offset = self.byte_offset_for(&src_indices);
                unsafe { ops.copy_element(src_ptr, src_offset, result_ptr, dst_idx); }
                dst_idx += 1;
            }
            increment_indices(&mut src_indices, self.shape());
        }

        Some(result)
    }

    /// Select elements/rows by integer indices. Returns a new array.
    /// For 1D array: selects elements at given indices.
    /// For ND array: selects along axis 0 (rows for 2D).
    pub fn select_by_indices(&self, indices: &RumpyArray) -> Option<Self> {
        let num_indices = indices.size();

        // Output shape: replace first dimension with number of indices
        let mut out_shape = self.shape().to_vec();
        if out_shape.is_empty() {
            return None; // Can't index a scalar
        }
        out_shape[0] = num_indices;

        let mut result = Self::zeros(out_shape, self.dtype());
        if num_indices == 0 {
            return Some(result);
        }

        let axis_len = self.shape()[0];
        let slice_size: usize = self.shape()[1..].iter().product::<usize>().max(1);

        let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
        let result_ptr = buffer.as_mut_ptr();
        let ops = result.dtype.ops();

        let src_ptr = self.data_ptr();
        let mut idx_indices = vec![0usize; indices.ndim()];
        for i in 0..num_indices {
            let idx_val = indices.get_element(&idx_indices) as isize;
            let idx = if idx_val < 0 {
                (axis_len as isize + idx_val) as usize
            } else {
                idx_val as usize
            };

            if idx >= axis_len {
                return None; // Index out of bounds
            }

            // Copy the slice at this index
            let mut src_indices = vec![0usize; self.ndim()];
            src_indices[0] = idx;

            for j in 0..slice_size {
                let src_offset = self.byte_offset_for(&src_indices);
                unsafe { ops.copy_element(src_ptr, src_offset, result_ptr, i * slice_size + j); }

                // Increment indices for dimensions 1..
                for d in (1..self.ndim()).rev() {
                    src_indices[d] += 1;
                    if src_indices[d] < self.shape()[d] {
                        break;
                    }
                    src_indices[d] = 0;
                }
            }

            increment_indices(&mut idx_indices, indices.shape());
        }

        Some(result)
    }

    /// Create a contiguous copy of the array.
    pub fn copy(&self) -> Self {
        self.astype(self.dtype())
    }

    /// Convert array to a new dtype.
    pub fn astype(&self, new_dtype: DType) -> Self {
        let mut result = Self::zeros(self.shape().to_vec(), new_dtype.clone());
        let size = self.size();
        if size == 0 {
            return result;
        }

        let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
        let result_ptr = buffer.as_mut_ptr();
        let dst_ops = result.dtype.ops();
        let src_ptr = self.data_ptr();
        let src_ops = self.dtype.ops();

        // Fast path: contiguous same-dtype copy
        if self.dtype == new_dtype && self.is_c_contiguous() {
            let nbytes = size * self.itemsize();
            unsafe { std::ptr::copy_nonoverlapping(src_ptr, result_ptr, nbytes); }
            return result;
        }

        // Fast path for broadcast copy (common in meshgrid/indices)
        if self.dtype == new_dtype && new_dtype == DType::float64() {
            self.copy_broadcast_typed::<f64>(result_ptr as *mut f64);
            return result;
        }
        if self.dtype == new_dtype && new_dtype == DType::int64() {
            self.copy_broadcast_typed::<i64>(result_ptr as *mut i64);
            return result;
        }

        if self.dtype == new_dtype {
            // Same dtype: copy elements directly
            for (i, offset) in self.iter_offsets().enumerate() {
                unsafe { dst_ops.copy_element(src_ptr, offset, result_ptr, i); }
            }
        } else {
            // Cross-dtype conversion via f64 (lossy for complex)
            for (i, offset) in self.iter_offsets().enumerate() {
                let val = unsafe { src_ops.read_f64(src_ptr, offset) }.unwrap_or(0.0);
                unsafe { dst_ops.write_f64(result_ptr, i, val); }
            }
        }
        result
    }

    /// Efficient copy for broadcast arrays - exploits zero strides.
    fn copy_broadcast_typed<T: Copy>(&self, dst: *mut T) {
        let shape = self.shape();
        let strides = self.strides();
        let src = self.data_ptr() as *const T;
        let ndim = shape.len();
        let size = self.size();
        let elem_size = std::mem::size_of::<T>() as isize;

        if size == 0 {
            return;
        }

        // For 2D broadcast (common case: meshgrid), use optimized path
        if ndim == 2 {
            let (n0, n1) = (shape[0], shape[1]);
            let (s0, s1) = (strides[0], strides[1]);

            if s0 == 0 && s1 != 0 {
                // Row broadcasts: same row repeated n0 times
                // First fill row 0, then memcpy to other rows
                let s1_elems = s1 / elem_size;
                for j in 0..n1 {
                    unsafe { *dst.add(j) = *src.offset(s1_elems * j as isize); }
                }
                for i in 1..n0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(dst, dst.add(i * n1), n1);
                    }
                }
                return;
            }

            if s0 != 0 && s1 == 0 {
                // Column broadcasts: same column repeated n1 times
                let s0_elems = s0 / elem_size;
                for i in 0..n0 {
                    let val = unsafe { *src.offset(s0_elems * i as isize) };
                    let row_start = i * n1;
                    for j in 0..n1 {
                        unsafe { *dst.add(row_start + j) = val; }
                    }
                }
                return;
            }
        }

        // Fallback: iterate with offset calculation
        for (i, offset) in self.iter_offsets().enumerate() {
            let byte_offset = offset / elem_size;
            unsafe { *dst.add(i) = *src.offset(byte_offset); }
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
pub(crate) fn is_c_contiguous(shape: &[usize], strides: &[isize], itemsize: usize) -> bool {
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
pub(crate) fn is_f_contiguous(shape: &[usize], strides: &[isize], itemsize: usize) -> bool {
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
