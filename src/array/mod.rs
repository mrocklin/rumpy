pub mod buffer;
pub mod dtype;
pub mod flags;
pub mod iter;

use std::sync::Arc;

pub use buffer::ArrayBuffer;
pub use dtype::{promote_dtype, DType, DTypeOps};
pub use flags::ArrayFlags;
pub use iter::{StridedIter, AxisOffsetIter};

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
            unsafe { ops.write_f64(ptr, i, v); }
        }
        arr
    }

    /// Create array from Vec with given shape.
    pub fn from_vec_with_shape(data: Vec<f64>, shape: Vec<usize>, dtype: DType) -> Self {
        let arr = Self::from_vec(data, dtype);
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
            unsafe { ops.write_f64(ptr, i, v as f64); }
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

    /// Create a view with new shape, strides, and dtype, sharing the buffer.
    pub fn view_with_dtype(&self, shape: Vec<usize>, strides: Vec<isize>, dtype: DType) -> Self {
        let mut flags = ArrayFlags::WRITEABLE;
        if is_c_contiguous(&shape, &strides, dtype.itemsize()) {
            flags |= ArrayFlags::C_CONTIGUOUS;
        }
        if is_f_contiguous(&shape, &strides, dtype.itemsize()) {
            flags |= ArrayFlags::F_CONTIGUOUS;
        }

        Self {
            buffer: Arc::clone(&self.buffer),
            offset: self.offset,
            shape,
            strides,
            dtype,
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
        } else if start > stop { (start - stop - step - 1) / (-step) } else { 0 };
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

    /// Transpose with specified axis order. Returns a view.
    pub fn transpose_axes(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.ndim(), "axes must have same length as ndim");
        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let new_strides: Vec<isize> = axes.iter().map(|&a| self.strides[a]).collect();
        self.view_with(0, new_shape, new_strides)
    }

    /// Flip array along given axis. Returns a view.
    pub fn flip(&self, axis: usize) -> Option<Self> {
        if axis >= self.ndim() {
            return None;
        }
        let len = self.shape[axis] as isize;
        if len == 0 {
            return Some(self.clone());
        }
        // slice from len-1 to -1 (exclusive) with step -1
        Some(self.slice_axis(axis, len - 1, -1, -1))
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

    /// Remove single-dimensional entries from the shape.
    /// Returns a view.
    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape().iter().copied().filter(|&d| d != 1).collect();
        let new_strides: Vec<isize> = self.shape().iter().zip(self.strides())
            .filter(|(&d, _)| d != 1)
            .map(|(_, &s)| s)
            .collect();

        // Handle scalar case (all dims were 1)
        if new_shape.is_empty() && self.size() == 1 {
            return self.view_with(0, vec![1], vec![self.itemsize() as isize]);
        }

        self.view_with(0, new_shape, new_strides)
    }

    /// Insert a new axis at the given position.
    /// Returns a view.
    pub fn expand_dims(&self, axis: usize) -> Option<Self> {
        if axis > self.ndim() {
            return None;
        }

        let mut new_shape = self.shape().to_vec();
        let mut new_strides = self.strides().to_vec();

        new_shape.insert(axis, 1);
        // Stride for size-1 dimension doesn't matter, but we use itemsize for consistency
        new_strides.insert(axis, self.itemsize() as isize);

        Some(self.view_with(0, new_shape, new_strides))
    }

    /// Remove a single-dimensional axis at the given position.
    /// Returns a view. If the axis doesn't have size 1, returns self unchanged.
    pub fn squeeze_axis(&self, axis: usize) -> Self {
        if axis >= self.ndim() || self.shape()[axis] != 1 {
            return self.clone();
        }

        let mut new_shape = self.shape().to_vec();
        let mut new_strides = self.strides().to_vec();

        new_shape.remove(axis);
        new_strides.remove(axis);

        // Handle scalar case
        if new_shape.is_empty() {
            return self.view_with(0, vec![], vec![]);
        }

        self.view_with(0, new_shape, new_strides)
    }

    /// Extract core sub-array at loop_indices for gufunc operations.
    ///
    /// Given an array with shape [L0, L1, ..., C0, C1, ...] where
    /// L* are loop dims and C* are core dims, extract the core
    /// sub-array at position loop_indices in the loop dimensions.
    ///
    /// Handles broadcasting: if a loop dimension has size 1, index 0 is used.
    pub fn gufunc_subarray(&self, loop_indices: &[usize], loop_ndim: usize) -> Self {
        let my_loop_ndim = loop_ndim.min(self.ndim());

        // Compute byte offset for this position in loop dims
        let mut offset_delta: isize = 0;
        for (i, &idx) in loop_indices.iter().enumerate() {
            if i < my_loop_ndim {
                // Handle broadcasting: if loop dim is 1, use index 0
                let dim_size = self.shape[i];
                let actual_idx = if dim_size == 1 { 0 } else { idx };
                offset_delta += (actual_idx as isize) * self.strides[i];
            }
        }

        // Core shape and strides are the trailing dimensions
        let core_shape = self.shape[my_loop_ndim..].to_vec();
        let core_strides = self.strides[my_loop_ndim..].to_vec();

        // Handle case where we're extracting a scalar (empty core dims)
        if core_shape.is_empty() {
            return self.view_with(offset_delta.max(0) as usize, vec![1], vec![self.itemsize() as isize]);
        }

        self.view_with(offset_delta.max(0) as usize, core_shape, core_strides)
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

    // === Formatting for repr/str ===

    /// Format array as repr string: `array([...])`.
    pub fn format_repr(&self) -> String {
        use dtype::DTypeKind;

        let threshold = 1000;
        let edgeitems = 3;
        let truncate = self.size() > threshold;

        let data = self.format_data(", ", truncate, edgeitems, 7); // 7 = len("array([")

        // Determine if dtype suffix is needed
        let needs_dtype = !matches!(self.dtype.kind(), DTypeKind::Int64 | DTypeKind::Float64 | DTypeKind::Bool);

        // Show shape for truncated arrays (numpy shows shape for any truncated array)
        let show_shape = truncate;

        // Format shape as tuple: (3, 4) not [3, 4]
        let shape_str = if self.ndim() == 1 {
            format!("({},)", self.shape[0])
        } else {
            format!("({})", self.shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "))
        };

        // Empty arrays always show dtype
        let suffix = if self.size() == 0 {
            format!(", dtype={}", self.dtype.ops().name())
        } else if show_shape && needs_dtype {
            format!(", shape={}, dtype={}", shape_str, self.dtype.ops().name())
        } else if show_shape {
            format!(", shape={}", shape_str)
        } else if needs_dtype {
            format!(", dtype={}", self.dtype.ops().name())
        } else {
            String::new()
        };

        format!("array({}{})", data, suffix)
    }

    /// Format array as str: `[...]` without wrapper.
    pub fn format_str(&self) -> String {
        let threshold = 1000;
        let edgeitems = 3;
        let truncate = self.size() > threshold;

        self.format_data(" ", truncate, edgeitems, 1) // 1 = len("[")
    }

    /// Format the data portion with given separator.
    /// base_indent is extra indentation for multi-line arrays (e.g., 7 for "array([").
    fn format_data(&self, sep: &str, truncate: bool, edgeitems: usize, base_indent: usize) -> String {
        if self.size() == 0 {
            return "[]".to_string();
        }

        // Collect all element strings for width calculation
        let elements = self.collect_elements(truncate, edgeitems);

        // Calculate max width for alignment
        let max_width = elements.iter().map(|s| s.len()).max().unwrap_or(0);

        // Format recursively
        self.format_recursive(&[], sep, truncate, edgeitems, max_width, base_indent)
    }

    /// Get indices to display for a dimension, with truncation.
    /// Returns (indices, needs_ellipsis).
    fn display_indices(&self, dim_size: usize, truncate: bool, edgeitems: usize) -> (Vec<usize>, bool) {
        if truncate && dim_size > 2 * edgeitems {
            let indices = (0..edgeitems).chain((dim_size - edgeitems)..dim_size).collect();
            (indices, true)
        } else {
            ((0..dim_size).collect(), false)
        }
    }

    /// Collect element strings (for width calculation).
    fn collect_elements(&self, truncate: bool, edgeitems: usize) -> Vec<String> {
        let mut result = Vec::new();
        self.collect_elements_recursive(&mut result, &[], truncate, edgeitems);
        result
    }

    fn collect_elements_recursive(
        &self,
        result: &mut Vec<String>,
        prefix: &[usize],
        truncate: bool,
        edgeitems: usize,
    ) {
        let depth = prefix.len();
        let (indices, _) = self.display_indices(self.shape[depth], truncate, edgeitems);
        let ptr = self.data_ptr();

        for i in indices {
            let mut new_prefix = prefix.to_vec();
            new_prefix.push(i);

            if new_prefix.len() == self.ndim() {
                let offset = self.byte_offset_for(&new_prefix);
                result.push(unsafe { self.dtype.ops().format_element(ptr, offset) });
            } else {
                self.collect_elements_recursive(result, &new_prefix, truncate, edgeitems);
            }
        }
    }

    /// Recursive formatting with proper indentation.
    fn format_recursive(
        &self,
        prefix: &[usize],
        sep: &str,
        truncate: bool,
        edgeitems: usize,
        width: usize,
        base_indent: usize,
    ) -> String {
        let depth = prefix.len();

        if depth == self.ndim() {
            let ptr = self.data_ptr();
            let offset = self.byte_offset_for(prefix);
            let s = unsafe { self.dtype.ops().format_element(ptr, offset) };
            return format!("{:>w$}", s, w = width);
        }

        let dim_size = self.shape[depth];
        let is_last_dim = depth == self.ndim() - 1;
        let (indices, has_ellipsis) = self.display_indices(dim_size, truncate, edgeitems);

        let mut parts = Vec::new();
        for (idx_pos, i) in indices.into_iter().enumerate() {
            // Insert ellipsis between front and back sections
            if has_ellipsis && idx_pos == edgeitems {
                parts.push("...".to_string());
            }

            let mut new_prefix = prefix.to_vec();
            new_prefix.push(i);
            parts.push(self.format_recursive(&new_prefix, sep, truncate, edgeitems, width, base_indent));
        }

        if is_last_dim {
            format!("[{}]", parts.join(sep))
        } else {
            // Multi-line formatting for higher dimensions
            let indent = " ".repeat(base_indent + depth);
            let inner_sep = if depth == self.ndim() - 2 {
                format!("{}\n{}", sep.trim_end(), indent)
            } else {
                // Extra blank line between highest-level blocks
                format!("{}\n\n{}", sep.trim_end(), indent)
            };
            format!("[{}]", parts.join(&inner_sep))
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

/// Concatenate arrays along an axis.
/// All arrays must have same shape except in the concatenation axis.
pub fn concatenate(arrays: &[RumpyArray], axis: usize) -> Option<RumpyArray> {
    if arrays.is_empty() {
        return None;
    }

    let first = &arrays[0];
    let ndim = first.ndim();

    if axis >= ndim {
        return None;
    }

    // Check all arrays have same shape except for concat axis
    for arr in arrays.iter().skip(1) {
        if arr.ndim() != ndim {
            return None;
        }
        for (i, (&a, &b)) in first.shape().iter().zip(arr.shape().iter()).enumerate() {
            if i != axis && a != b {
                return None;
            }
        }
    }

    // Compute output shape
    let mut out_shape = first.shape().to_vec();
    out_shape[axis] = arrays.iter().map(|a| a.shape()[axis]).sum();

    // Use dtype of first array (TODO: dtype promotion)
    let dtype = first.dtype();
    let mut result = RumpyArray::zeros(out_shape, dtype);

    if result.size() == 0 {
        return Some(result);
    }

    let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
    let result_ptr = buffer.as_mut_ptr();
    let ops = result.dtype.ops();

    // Copy data from each array
    let mut axis_offset = 0usize;
    for arr in arrays {
        let size = arr.size();
        let src_ptr = arr.data_ptr();
        let mut src_indices = vec![0usize; ndim];
        for _ in 0..size {
            // Compute destination indices
            let mut dst_indices = src_indices.clone();
            dst_indices[axis] += axis_offset;

            // Compute linear index in result
            let mut dst_idx = 0;
            let mut stride = 1;
            for i in (0..ndim).rev() {
                dst_idx += dst_indices[i] * stride;
                stride *= result.shape()[i];
            }

            let src_offset = arr.byte_offset_for(&src_indices);
            unsafe { ops.copy_element(src_ptr, src_offset, result_ptr, dst_idx); }

            increment_indices(&mut src_indices, arr.shape());
        }
        axis_offset += arr.shape()[axis];
    }

    Some(result)
}

/// Stack arrays along a new axis.
/// All arrays must have the same shape.
pub fn stack(arrays: &[RumpyArray], axis: usize) -> Option<RumpyArray> {
    if arrays.is_empty() {
        return None;
    }

    let first = &arrays[0];

    // axis can be at most ndim (to add at end)
    if axis > first.ndim() {
        return None;
    }

    // Check all arrays have same shape
    for arr in arrays.iter().skip(1) {
        if arr.shape() != first.shape() {
            return None;
        }
    }

    // Expand dims on each array, then concatenate
    let expanded: Vec<RumpyArray> = arrays
        .iter()
        .filter_map(|a| a.expand_dims(axis))
        .collect();

    if expanded.len() != arrays.len() {
        return None;
    }

    concatenate(&expanded, axis)
}

/// Split array into equal parts along axis.
/// Returns None if array cannot be split evenly.
pub fn split(arr: &RumpyArray, num_sections: usize, axis: usize) -> Option<Vec<RumpyArray>> {
    if num_sections == 0 || axis >= arr.ndim() {
        return None;
    }

    let axis_len = arr.shape()[axis];
    if !axis_len.is_multiple_of(num_sections) {
        return None; // Must divide evenly
    }

    let section_size = axis_len / num_sections;
    Some(split_into_sizes(arr, &vec![section_size; num_sections], axis))
}

/// Split array into sections, allowing unequal sizes.
/// For `n % num_sections` leftover elements, first sections get one extra.
pub fn array_split(arr: &RumpyArray, num_sections: usize, axis: usize) -> Option<Vec<RumpyArray>> {
    if num_sections == 0 || axis >= arr.ndim() {
        return None;
    }

    let axis_len = arr.shape()[axis];
    let base_size = axis_len / num_sections;
    let remainder = axis_len % num_sections;

    // First `remainder` sections get base_size + 1, rest get base_size
    let sizes: Vec<usize> = (0..num_sections)
        .map(|i| if i < remainder { base_size + 1 } else { base_size })
        .collect();

    Some(split_into_sizes(arr, &sizes, axis))
}

/// Helper to split array into sections with given sizes along axis.
/// Returns views (no copy) like numpy.
fn split_into_sizes(arr: &RumpyArray, sizes: &[usize], axis: usize) -> Vec<RumpyArray> {
    let mut result = Vec::with_capacity(sizes.len());
    let mut start = 0isize;

    for &size in sizes {
        let section = arr.slice_axis(axis, start, start + size as isize, 1);
        result.push(section); // Return view, not copy
        start += size as isize;
    }

    result
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

/// Count occurrences of each value in an array of non-negative integers.
/// Returns an array of counts where result[i] = number of times i appears in x.
pub fn bincount(x: &RumpyArray, minlength: usize) -> Option<RumpyArray> {
    // Must be 1D
    if x.ndim() != 1 {
        return None;
    }

    let size = x.size();
    if size == 0 {
        return Some(RumpyArray::zeros(vec![minlength.max(0)], DType::int64()));
    }

    let ptr = x.data_ptr();
    let dtype = x.dtype();
    let ops = dtype.ops();

    // Find max value to determine output size
    let mut max_val: i64 = 0;
    for offset in x.iter_offsets() {
        let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
        let ival = val as i64;
        if ival < 0 {
            return None; // Negative values not allowed
        }
        if ival > max_val {
            max_val = ival;
        }
    }

    let out_len = (max_val as usize + 1).max(minlength);
    let mut counts = vec![0i64; out_len];

    // Count occurrences
    for offset in x.iter_offsets() {
        let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
        let idx = val as usize;
        counts[idx] += 1;
    }

    // Convert to RumpyArray
    let mut result = RumpyArray::zeros(vec![out_len], DType::int64());
    let buffer = result.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut i64;
    for (i, &count) in counts.iter().enumerate() {
        unsafe { *result_ptr.add(i) = count; }
    }

    Some(result)
}

/// Compute the q-th percentile(s) of the data.
/// q values should be in [0, 100].
/// Uses linear interpolation (numpy default method).
pub fn percentile(arr: &RumpyArray, q: &[f64], axis: Option<usize>) -> Option<RumpyArray> {
    match axis {
        None => percentile_flat(arr, q),
        Some(ax) => percentile_axis(arr, q, ax),
    }
}

/// Percentile over flattened array.
fn percentile_flat(arr: &RumpyArray, q: &[f64]) -> Option<RumpyArray> {
    let size = arr.size();
    if size == 0 {
        return Some(RumpyArray::zeros(vec![q.len()], DType::float64()));
    }

    // Collect and sort values
    let ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();

    let mut values: Vec<f64> = Vec::with_capacity(size);
    for offset in arr.iter_offsets() {
        values.push(unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0));
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute percentiles using linear interpolation
    let mut result = RumpyArray::zeros(vec![q.len()], DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    for (i, &pct) in q.iter().enumerate() {
        let val = interpolate_percentile(&values, pct);
        unsafe { *result_ptr.add(i) = val; }
    }

    Some(result)
}

/// Percentile along axis.
fn percentile_axis(arr: &RumpyArray, q: &[f64], axis: usize) -> Option<RumpyArray> {
    if axis >= arr.ndim() {
        return None;
    }

    let shape = arr.shape();
    let axis_len = shape[axis];
    let axis_stride = arr.strides()[axis];

    // Output shape: remove axis, prepend q.len() if multiple percentiles
    let mut reduced_shape: Vec<usize> = shape[..axis].to_vec();
    reduced_shape.extend_from_slice(&shape[axis + 1..]);
    if reduced_shape.is_empty() {
        reduced_shape = vec![1];
    }

    let out_shape = if q.len() == 1 {
        reduced_shape.clone()
    } else {
        let mut s = vec![q.len()];
        s.extend(&reduced_shape);
        s
    };

    let reduced_size: usize = reduced_shape.iter().product();
    if axis_len == 0 || reduced_size == 0 {
        return Some(RumpyArray::zeros(out_shape, DType::float64()));
    }

    let mut result = RumpyArray::zeros(out_shape, DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    let src_ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();

    // Use axis_offsets iterator (same pattern as reduce_axis_op)
    for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
        // Collect values along axis from this base offset
        let mut values: Vec<f64> = Vec::with_capacity(axis_len);
        let mut ptr = unsafe { src_ptr.offset(base_offset) };
        for _ in 0..axis_len {
            values.push(unsafe { ops.read_f64(ptr, 0) }.unwrap_or(0.0));
            ptr = unsafe { ptr.offset(axis_stride) };
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute percentiles
        if q.len() == 1 {
            unsafe { *result_ptr.add(i) = interpolate_percentile(&values, q[0]); }
        } else {
            for (qi, &pct) in q.iter().enumerate() {
                unsafe { *result_ptr.add(qi * reduced_size + i) = interpolate_percentile(&values, pct); }
            }
        }
    }

    Some(result)
}

/// Linear interpolation for percentile.
fn interpolate_percentile(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }

    // Convert percentile to index (numpy's linear interpolation method)
    let idx = (pct / 100.0) * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;

    if lo == hi || hi >= n {
        sorted[lo.min(n - 1)]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Create logarithmically spaced array.
/// Returns base^linspace(start, stop, num).
pub fn logspace(start: f64, stop: f64, num: usize, base: f64, dtype: DType) -> RumpyArray {
    let mut arr = RumpyArray::zeros(vec![num], dtype);
    if num == 0 {
        return arr;
    }

    let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
    let ptr = buffer.as_mut_ptr();
    let ops = arr.dtype.ops();

    let step = if num > 1 { (stop - start) / (num - 1) as f64 } else { 0.0 };

    for i in 0..num {
        let exponent = start + (i as f64) * step;
        let val = base.powf(exponent);
        unsafe { ops.write_f64(ptr, i, val); }
    }
    arr
}

/// Create geometrically spaced array.
/// Returns numbers spaced evenly on a log scale between start and stop.
pub fn geomspace(start: f64, stop: f64, num: usize, dtype: DType) -> Option<RumpyArray> {
    // start and stop must have same sign (and not zero)
    if start * stop <= 0.0 {
        return None;
    }

    let mut arr = RumpyArray::zeros(vec![num], dtype);
    if num == 0 {
        return Some(arr);
    }

    let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
    let ptr = buffer.as_mut_ptr();
    let ops = arr.dtype.ops();

    // Handle negative values by using absolute values and restoring sign
    let sign = if start < 0.0 { -1.0 } else { 1.0 };
    let log_start = start.abs().ln();
    let log_stop = stop.abs().ln();
    let step = if num > 1 { (log_stop - log_start) / (num - 1) as f64 } else { 0.0 };

    for i in 0..num {
        let val = sign * (log_start + (i as f64) * step).exp();
        unsafe { ops.write_f64(ptr, i, val); }
    }
    Some(arr)
}

/// Create a triangular matrix of ones.
/// k: diagonal offset (0 = main diagonal, positive = above, negative = below)
pub fn tri(n: usize, m: usize, k: isize, dtype: DType) -> RumpyArray {
    let mut arr = RumpyArray::zeros(vec![n, m], dtype);
    if n == 0 || m == 0 {
        return arr;
    }

    let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
    let ptr = buffer.as_mut_ptr();
    let ops = arr.dtype.ops();

    for i in 0..n {
        for j in 0..m {
            // Element is 1 if j <= i + k
            if (j as isize) <= (i as isize) + k {
                let idx = i * m + j;
                unsafe { ops.write_one(ptr, idx); }
            }
        }
    }
    arr
}

/// Helper for tril/triu - extract triangle from 2D array.
fn extract_triangle(arr: &RumpyArray, k: isize, lower: bool) -> Option<RumpyArray> {
    if arr.ndim() != 2 {
        return None;
    }

    let shape = arr.shape();
    let n = shape[0];
    let m = shape[1];
    let mut result = RumpyArray::zeros(shape.to_vec(), arr.dtype());

    if n == 0 || m == 0 {
        return Some(result);
    }

    let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
    let result_ptr = buffer.as_mut_ptr();
    let itemsize = arr.itemsize();

    // Fast path: contiguous source array - use memcpy for row segments
    if arr.is_c_contiguous() {
        let src_ptr = arr.data_ptr();
        for i in 0..n {
            // Calculate range of columns to copy for this row
            let (start_col, end_col) = if lower {
                // Lower triangle: copy columns 0..min(i+k+1, m)
                let end = ((i as isize + k + 1).max(0) as usize).min(m);
                (0, end)
            } else {
                // Upper triangle: copy columns max(i+k, 0)..m
                let start = ((i as isize + k).max(0) as usize).min(m);
                (start, m)
            };

            if end_col > start_col {
                let src_row = unsafe { src_ptr.add(i * m * itemsize + start_col * itemsize) };
                let dst_row = unsafe { result_ptr.add(i * m * itemsize + start_col * itemsize) };
                let copy_bytes = (end_col - start_col) * itemsize;
                unsafe { std::ptr::copy_nonoverlapping(src_row, dst_row, copy_bytes); }
            }
        }
        return Some(result);
    }

    // Slow path: strided array - element by element
    let ops = result.dtype.ops();
    let src_ptr = arr.data_ptr();

    for i in 0..n {
        for j in 0..m {
            let include = if lower {
                (j as isize) <= (i as isize) + k
            } else {
                (j as isize) >= (i as isize) + k
            };
            if include {
                let offset = arr.byte_offset_for(&[i, j]);
                let idx = i * m + j;
                unsafe { ops.copy_element(src_ptr, offset, result_ptr, idx); }
            }
        }
    }
    Some(result)
}

/// Return lower triangle of an array.
pub fn tril(arr: &RumpyArray, k: isize) -> Option<RumpyArray> {
    extract_triangle(arr, k, true)
}

/// Return upper triangle of an array.
pub fn triu(arr: &RumpyArray, k: isize) -> Option<RumpyArray> {
    extract_triangle(arr, k, false)
}

/// Create a 2D array with flattened input as diagonal.
/// k: diagonal offset (0 = main diagonal)
pub fn diagflat(v: &RumpyArray, k: isize) -> RumpyArray {
    // Flatten input
    let size = v.size();
    let n = size + k.unsigned_abs();

    let mut result = RumpyArray::zeros(vec![n, n], v.dtype());
    if size == 0 {
        return result;
    }

    let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
    let result_ptr = buffer.as_mut_ptr();
    let ops = result.dtype.ops();
    let src_ptr = v.data_ptr();

    for (i, offset) in v.iter_offsets().enumerate() {
        let (row, col) = if k >= 0 {
            (i, i + k as usize)
        } else {
            (i + (-k) as usize, i)
        };
        let idx = row * n + col;
        unsafe { ops.copy_element(src_ptr, offset, result_ptr, idx); }
    }
    result
}

/// Return coordinate matrices from coordinate vectors.
/// indexing: "xy" (Cartesian) or "ij" (matrix)
pub fn meshgrid(arrays: &[RumpyArray], indexing: &str) -> Option<Vec<RumpyArray>> {
    if arrays.is_empty() {
        return Some(Vec::new());
    }

    // All inputs must be 1D
    for arr in arrays {
        if arr.ndim() != 1 {
            return None;
        }
    }

    let ndim = arrays.len();
    let sizes: Vec<usize> = arrays.iter().map(|a| a.size()).collect();

    // Output shape depends on indexing mode
    let output_shape: Vec<usize> = if indexing == "xy" && ndim >= 2 {
        let mut shape = sizes.clone();
        shape.swap(0, 1);
        shape
    } else {
        sizes.clone()
    };

    let mut result = Vec::with_capacity(ndim);

    for (dim, arr) in arrays.iter().enumerate() {
        // The dimension this array varies along in output
        let vary_dim = if indexing == "xy" && ndim >= 2 {
            if dim == 0 { 1 } else if dim == 1 { 0 } else { dim }
        } else {
            dim
        };

        // Create broadcast shape: all 1s except for vary_dim
        let mut bc_shape = vec![1usize; ndim];
        bc_shape[vary_dim] = arr.size();

        // Reshape to broadcast shape, then broadcast to output shape
        let reshaped = arr.reshape(bc_shape)?;
        let broadcast = reshaped.broadcast_to(&output_shape)?;

        // Copy to contiguous
        result.push(broadcast.copy());
    }

    Some(result)
}

/// Return an array representing indices of a grid.
/// Result shape is (len(dimensions),) + dimensions.
pub fn indices(dimensions: &[usize], dtype: DType) -> RumpyArray {
    let ndim = dimensions.len();
    if ndim == 0 {
        return RumpyArray::zeros(vec![0], dtype);
    }

    let mut shape = vec![ndim];
    shape.extend_from_slice(dimensions);

    let mut result = RumpyArray::zeros(shape.clone(), dtype.clone());
    let total_per_dim: usize = dimensions.iter().product();

    if total_per_dim == 0 {
        return result;
    }

    let buffer = Arc::get_mut(&mut result.buffer).expect("buffer must be unique");
    let ptr = buffer.as_mut_ptr();

    // Fast path for int64 (default dtype for indices)
    if dtype == DType::int64() {
        let dst = ptr as *mut i64;

        for dim in 0..ndim {
            let base = dim * total_per_dim;
            // For dimension dim, the index repeats with period = product of dims after it
            // and increments after every (product of dims after it) elements
            let stride: usize = dimensions[dim + 1..].iter().product();
            let repeat: usize = dimensions[..dim].iter().product();
            let dim_size = dimensions[dim];

            for r in 0..repeat {
                let r_base = r * dim_size * stride;
                for idx in 0..dim_size {
                    let val = idx as i64;
                    for s in 0..stride {
                        unsafe { *dst.add(base + r_base + idx * stride + s) = val; }
                    }
                }
            }
        }
    } else if dtype == DType::float64() {
        let dst = ptr as *mut f64;

        for dim in 0..ndim {
            let base = dim * total_per_dim;
            let stride: usize = dimensions[dim + 1..].iter().product();
            let repeat: usize = dimensions[..dim].iter().product();
            let dim_size = dimensions[dim];

            for r in 0..repeat {
                let r_base = r * dim_size * stride;
                for idx in 0..dim_size {
                    let val = idx as f64;
                    for s in 0..stride {
                        unsafe { *dst.add(base + r_base + idx * stride + s) = val; }
                    }
                }
            }
        }
    } else {
        // Fallback for other dtypes
        let ops = result.dtype.ops();
        for dim in 0..ndim {
            let base_offset = dim * total_per_dim;
            let mut idx_vec = vec![0usize; ndim];

            for i in 0..total_per_dim {
                let val = idx_vec[dim] as f64;
                unsafe { ops.write_f64(ptr, base_offset + i, val); }
                increment_indices(&mut idx_vec, dimensions);
            }
        }
    }

    result
}

/// 1D discrete convolution.
/// mode: "full", "same", or "valid"
pub fn convolve(a: &RumpyArray, v: &RumpyArray, mode: &str) -> Option<RumpyArray> {
    // Both must be 1D
    if a.ndim() != 1 || v.ndim() != 1 {
        return None;
    }

    let n = a.size();
    let m = v.size();

    if n == 0 || m == 0 {
        let out_len = match mode {
            "full" => n + m - 1,
            "same" => n,
            "valid" => n.saturating_sub(m - 1),
            _ => return None,
        };
        return Some(RumpyArray::zeros(vec![out_len.max(0)], DType::float64()));
    }

    // Collect values
    let a_ptr = a.data_ptr();
    let a_dtype = a.dtype();
    let a_ops = a_dtype.ops();
    let v_ptr = v.data_ptr();
    let v_dtype = v.dtype();
    let v_ops = v_dtype.ops();

    let mut a_vals: Vec<f64> = Vec::with_capacity(n);
    for offset in a.iter_offsets() {
        a_vals.push(unsafe { a_ops.read_f64(a_ptr, offset) }.unwrap_or(0.0));
    }

    let mut v_vals: Vec<f64> = Vec::with_capacity(m);
    for offset in v.iter_offsets() {
        v_vals.push(unsafe { v_ops.read_f64(v_ptr, offset) }.unwrap_or(0.0));
    }

    // Output size depends on mode
    let (out_len, offset) = match mode {
        "full" => (n + m - 1, 0isize),
        "same" => (n, (m as isize - 1) / 2),
        "valid" => (n.saturating_sub(m - 1).max(1), m as isize - 1),
        _ => return None,
    };

    let mut result = RumpyArray::zeros(vec![out_len], DType::float64());
    let buffer = result.buffer_mut();
    let result_buffer = std::sync::Arc::get_mut(buffer).expect("unique");
    let result_ptr = result_buffer.as_mut_ptr() as *mut f64;

    // Convolution: result[k] = sum_j a[k-j] * v[j] for valid indices
    // For "full" mode, k ranges from 0 to n+m-2
    for k in 0..out_len {
        let full_k = k as isize + offset;
        let mut sum = 0.0;
        for j in 0..m {
            let i = full_k - j as isize;
            if i >= 0 && (i as usize) < n {
                sum += a_vals[i as usize] * v_vals[j];
            }
        }
        unsafe { *result_ptr.add(k) = sum; }
    }

    Some(result)
}

// Stream 11: Array Manipulation Functions

/// Helper: compute linear index from multi-dimensional indices for C-contiguous array.
fn linear_index(indices: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for i in (0..indices.len()).rev() {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    idx
}

/// Repeat elements of an array.
/// If axis is None, flatten and repeat. Otherwise repeat along axis.
pub fn repeat(arr: &RumpyArray, repeats: usize, axis: Option<isize>) -> Option<RumpyArray> {
    if repeats == 0 {
        return Some(RumpyArray::zeros(vec![0], arr.dtype()));
    }

    match axis {
        None => {
            // Flatten and repeat each element
            let size = arr.size();
            let new_size = size * repeats;
            let src_dtype = arr.dtype();
            let mut result = RumpyArray::zeros(vec![new_size], src_dtype.clone());

            let src_ptr = arr.data_ptr();
            let src_ops = src_dtype.ops();
            let dst_dtype = result.dtype();
            let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
            let dst_ptr = buffer.as_mut_ptr();
            let dst_ops = dst_dtype.ops();

            let mut dst_idx = 0;
            for offset in arr.iter_offsets() {
                let val = unsafe { src_ops.read_f64(src_ptr, offset) }.unwrap_or(0.0);
                for _ in 0..repeats {
                    unsafe { dst_ops.write_f64(dst_ptr, dst_idx, val); }
                    dst_idx += 1;
                }
            }
            Some(result)
        }
        Some(ax) => {
            let ndim = arr.ndim();
            let axis = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            if axis >= ndim {
                return None;
            }

            // New shape with axis dimension multiplied by repeats
            let mut new_shape = arr.shape().to_vec();
            new_shape[axis] *= repeats;
            let src_dtype = arr.dtype();
            let mut result = RumpyArray::zeros(new_shape.clone(), src_dtype.clone());

            let src_ptr = arr.data_ptr();
            let src_ops = src_dtype.ops();
            let dst_dtype = result.dtype();
            let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
            let dst_ptr = buffer.as_mut_ptr();
            let dst_ops = dst_dtype.ops();

            // Iterate through source indices and write repeated values
            let src_shape = arr.shape();
            let mut src_indices = vec![0usize; ndim];
            let mut dst_indices = vec![0usize; ndim];

            let size = arr.size();
            for _ in 0..size {
                let src_offset = arr.byte_offset_for(&src_indices);
                let val = unsafe { src_ops.read_f64(src_ptr, src_offset) }.unwrap_or(0.0);

                // Write this value `repeats` times along the axis
                dst_indices.copy_from_slice(&src_indices);
                let base_axis_idx = src_indices[axis] * repeats;
                for r in 0..repeats {
                    dst_indices[axis] = base_axis_idx + r;
                    let dst_offset = linear_index(&dst_indices, &new_shape);
                    unsafe { dst_ops.write_f64(dst_ptr, dst_offset, val); }
                }

                increment_indices(&mut src_indices, src_shape);
            }

            Some(result)
        }
    }
}

/// Construct array by repeating input array by given reps.
pub fn tile(arr: &RumpyArray, reps: &[usize]) -> RumpyArray {
    if reps.is_empty() || reps.iter().all(|&r| r == 1) {
        return arr.copy();
    }

    // Extend reps to match dimensions if needed
    let arr_ndim = arr.ndim();
    let reps_len = reps.len();
    let max_ndim = arr_ndim.max(reps_len);

    // Pad reps with 1s on the left
    let mut full_reps = vec![1usize; max_ndim];
    for (i, &r) in reps.iter().enumerate() {
        full_reps[max_ndim - reps_len + i] = r;
    }

    // Pad shape with 1s on the left
    let src_shape = arr.shape();
    let mut padded_shape = vec![1usize; max_ndim];
    for (i, &s) in src_shape.iter().enumerate() {
        padded_shape[max_ndim - arr_ndim + i] = s;
    }

    // New shape
    let new_shape: Vec<usize> = padded_shape.iter().zip(full_reps.iter()).map(|(&s, &r)| s * r).collect();
    let src_dtype = arr.dtype();
    let mut result = RumpyArray::zeros(new_shape.clone(), src_dtype.clone());

    // Fast path: 1D contiguous array - use memcpy
    if arr_ndim == 1 && reps_len == 1 && arr.is_c_contiguous() {
        let src_size = arr.size();
        let itemsize = arr.itemsize();
        let nbytes = src_size * itemsize;
        let src_ptr = arr.data_ptr();
        let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
        let dst_ptr = buffer.as_mut_ptr();

        for i in 0..reps[0] {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr,
                    dst_ptr.add(i * nbytes),
                    nbytes
                );
            }
        }
        return result;
    }

    let src_ptr = arr.data_ptr();
    let src_ops = src_dtype.ops();
    let dst_dtype = result.dtype();
    let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
    let dst_ptr = buffer.as_mut_ptr();
    let dst_ops = dst_dtype.ops();

    // Iterate through destination indices
    let mut dst_indices = vec![0usize; max_ndim];
    let dst_size: usize = new_shape.iter().product();

    for i in 0..dst_size {
        // Map destination index to source index (modulo)
        let src_indices: Vec<usize> = dst_indices.iter()
            .zip(padded_shape.iter())
            .map(|(&di, &ps)| di % ps)
            .collect();

        // Get source offset (for original array shape)
        let actual_src_indices: Vec<usize> = if arr_ndim < max_ndim {
            // Use only the rightmost dimensions
            src_indices[max_ndim - arr_ndim..].to_vec()
        } else {
            src_indices
        };
        let src_byte_offset = arr.byte_offset_for(&actual_src_indices);
        let val = unsafe { src_ops.read_f64(src_ptr, src_byte_offset) }.unwrap_or(0.0);

        // Destination offset (result is C-contiguous, so use linear index)
        let dst_offset = linear_index(&dst_indices, &new_shape);
        unsafe { dst_ops.write_f64(dst_ptr, dst_offset, val); }

        // Increment destination indices
        if i + 1 < dst_size {
            increment_indices(&mut dst_indices, &new_shape);
        }
    }

    result
}

/// Helper: flatten array to 1D (always copies).
fn ravel_arr(arr: &RumpyArray) -> RumpyArray {
    let size = arr.size();
    if let Some(view) = arr.reshape(vec![size]) {
        view
    } else {
        arr.copy().reshape(vec![size]).unwrap()
    }
}

/// Append values to end of array.
/// If axis is None, both arrays are flattened before concatenation.
pub fn append(arr: &RumpyArray, values: &RumpyArray, axis: Option<isize>) -> Option<RumpyArray> {
    match axis {
        None => {
            // Flatten both and concatenate
            let flat_arr = ravel_arr(arr);
            let flat_vals = ravel_arr(values);
            concatenate(&[flat_arr, flat_vals], 0)
        }
        Some(ax) => {
            let ndim = arr.ndim();
            let axis = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            concatenate(&[arr.clone(), values.clone()], axis)
        }
    }
}

/// Insert values into array at given index.
/// If axis is None, flatten first.
pub fn insert(arr: &RumpyArray, index: isize, values: &RumpyArray, axis: Option<isize>) -> Option<RumpyArray> {
    match axis {
        None => {
            // Flatten
            let flat = ravel_arr(arr);
            let size = flat.size();
            let idx = if index < 0 { (size as isize + index + 1).max(0) as usize } else { index as usize };
            let idx = idx.min(size);

            // Split at index, insert, concatenate
            let before = if idx > 0 { flat.slice_axis(0, 0, idx as isize, 1) } else { RumpyArray::zeros(vec![0], arr.dtype()) };
            let after = if idx < size { flat.slice_axis(0, idx as isize, size as isize, 1) } else { RumpyArray::zeros(vec![0], arr.dtype()) };
            let flat_vals = ravel_arr(values);

            concatenate(&[before, flat_vals, after], 0)
        }
        Some(ax) => {
            let ndim = arr.ndim();
            let axis = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            if axis >= ndim {
                return None;
            }

            let axis_len = arr.shape()[axis];
            let idx = if index < 0 { (axis_len as isize + index + 1).max(0) as usize } else { index as usize };
            let idx = idx.min(axis_len);

            let before = arr.slice_axis(axis, 0, idx as isize, 1);
            let after = arr.slice_axis(axis, idx as isize, axis_len as isize, 1);

            // Values need to be reshaped to match
            let mut val_shape = arr.shape().to_vec();
            val_shape[axis] = values.size() / (arr.size() / axis_len).max(1);
            let vals_reshaped = values.reshape(val_shape).unwrap_or_else(|| values.clone());

            concatenate(&[before, vals_reshaped, after], axis)
        }
    }
}

/// Delete elements from array at given index.
/// If axis is None, flatten first.
pub fn delete(arr: &RumpyArray, index: isize, axis: Option<isize>) -> Option<RumpyArray> {
    match axis {
        None => {
            let flat = ravel_arr(arr);
            let size = flat.size();
            if size == 0 {
                return Some(flat);
            }
            let idx = if index < 0 { (size as isize + index).max(0) as usize } else { index as usize };
            if idx >= size {
                return Some(flat);
            }

            let before = if idx > 0 { flat.slice_axis(0, 0, idx as isize, 1) } else { RumpyArray::zeros(vec![0], arr.dtype()) };
            let after = if idx + 1 < size { flat.slice_axis(0, (idx + 1) as isize, size as isize, 1) } else { RumpyArray::zeros(vec![0], arr.dtype()) };

            concatenate(&[before, after], 0)
        }
        Some(ax) => {
            let ndim = arr.ndim();
            let axis = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            if axis >= ndim {
                return None;
            }

            let axis_len = arr.shape()[axis];
            if axis_len == 0 {
                return Some(arr.clone());
            }
            let idx = if index < 0 { (axis_len as isize + index).max(0) as usize } else { index as usize };
            if idx >= axis_len {
                return Some(arr.clone());
            }

            let before = arr.slice_axis(axis, 0, idx as isize, 1);
            let after = arr.slice_axis(axis, (idx + 1) as isize, axis_len as isize, 1);

            concatenate(&[before, after], axis)
        }
    }
}

/// Pad an array.
/// pad_width is a slice of (before, after) tuples for each dimension.
pub fn pad(arr: &RumpyArray, pad_width: &[(usize, usize)], mode: &str, constant_value: f64) -> Option<RumpyArray> {
    let ndim = arr.ndim();
    if pad_width.len() != ndim {
        return None;
    }

    // Calculate new shape
    let new_shape: Vec<usize> = arr.shape().iter()
        .zip(pad_width.iter())
        .map(|(&s, &(b, a))| s + b + a)
        .collect();

    match mode {
        "constant" => {
            let mut result = RumpyArray::full(new_shape.clone(), constant_value, arr.dtype());
            copy_region(arr, &mut result, pad_width, &new_shape);
            Some(result)
        }
        "edge" => {
            let src_dtype = arr.dtype();
            let mut result = RumpyArray::zeros(new_shape.clone(), src_dtype.clone());
            let src_ops = src_dtype.ops();
            let dst_dtype = result.dtype();
            let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
            let dst_ptr = buffer.as_mut_ptr();
            let dst_ops = dst_dtype.ops();

            let mut dst_indices = vec![0usize; ndim];
            let dst_size: usize = new_shape.iter().product();
            let src_ptr = arr.data_ptr();

            // Fill all cells with edge-clamped values
            for _ in 0..dst_size {
                let src_indices: Vec<usize> = dst_indices.iter()
                    .zip(pad_width.iter())
                    .zip(arr.shape().iter())
                    .map(|((&di, &(b, _)), &s)| {
                        if di < b { 0 }
                        else if di >= b + s { s.saturating_sub(1) }
                        else { di - b }
                    })
                    .collect();

                let src_byte_offset = arr.byte_offset_for(&src_indices);
                let val = unsafe { src_ops.read_f64(src_ptr, src_byte_offset) }.unwrap_or(0.0);
                let dst_offset = linear_index(&dst_indices, &new_shape);
                unsafe { dst_ops.write_f64(dst_ptr, dst_offset, val); }

                increment_indices(&mut dst_indices, &new_shape);
            }

            Some(result)
        }
        _ => None,
    }
}

/// Helper to copy source array into a padded destination.
fn copy_region(src: &RumpyArray, dst: &mut RumpyArray, pad_width: &[(usize, usize)], dst_shape: &[usize]) {
    let ndim = src.ndim();
    let src_shape = src.shape();

    let src_ptr = src.data_ptr();
    let src_dtype = src.dtype();
    let src_ops = src_dtype.ops();
    let dst_dtype = dst.dtype();
    let buffer = Arc::get_mut(dst.buffer_mut()).expect("unique");
    let dst_ptr = buffer.as_mut_ptr();
    let dst_ops = dst_dtype.ops();

    let mut src_indices = vec![0usize; ndim];
    let src_size = src.size();

    for _ in 0..src_size {
        // Calculate destination indices by adding padding
        let dst_indices: Vec<usize> = src_indices.iter()
            .zip(pad_width.iter())
            .map(|(&si, &(b, _))| si + b)
            .collect();

        let src_byte_offset = src.byte_offset_for(&src_indices);
        let val = unsafe { src_ops.read_f64(src_ptr, src_byte_offset) }.unwrap_or(0.0);

        let dst_offset = linear_index(&dst_indices, dst_shape);
        unsafe { dst_ops.write_f64(dst_ptr, dst_offset, val); }

        increment_indices(&mut src_indices, src_shape);
    }
}

/// Roll array elements along given axis.
/// If axis is None, rolls over flattened array but preserves original shape.
pub fn roll(arr: &RumpyArray, shift: isize, axis: Option<isize>) -> RumpyArray {
    match axis {
        None => {
            // Flatten, roll, reshape back to original shape
            let original_shape = arr.shape().to_vec();
            let flat = ravel_arr(arr);
            let size = flat.size();
            if size == 0 {
                return arr.copy();
            }

            // Normalize shift
            let shift = ((shift % size as isize) + size as isize) as usize % size;
            if shift == 0 {
                return arr.copy();
            }

            let src_dtype = flat.dtype();
            let mut result = RumpyArray::zeros(original_shape.clone(), src_dtype.clone());
            let src_ptr = flat.data_ptr();
            let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
            let dst_ptr = buffer.as_mut_ptr();

            // Fast path: use memcpy for the two parts
            // Result: [src[size-shift:], src[:size-shift]]
            let itemsize = src_dtype.itemsize();
            let split_point = size - shift; // Elements from this index go to the front
            let bytes_part1 = shift * itemsize;       // Elements shift..size go to dst[0..shift]
            let bytes_part2 = split_point * itemsize; // Elements 0..split_point go to dst[shift..]

            unsafe {
                // Copy elements [split_point..] to [0..shift]
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(split_point * itemsize),
                    dst_ptr,
                    bytes_part1,
                );
                // Copy elements [0..split_point] to [shift..]
                std::ptr::copy_nonoverlapping(
                    src_ptr,
                    dst_ptr.add(bytes_part1),
                    bytes_part2,
                );
            }

            result
        }
        Some(ax) => {
            let ndim = arr.ndim();
            let axis = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            if axis >= ndim {
                return arr.copy();
            }

            let axis_len = arr.shape()[axis];
            if axis_len == 0 {
                return arr.copy();
            }

            // Normalize shift
            let shift = ((shift % axis_len as isize) + axis_len as isize) as usize % axis_len;
            if shift == 0 {
                return arr.copy();
            }

            let result_shape = arr.shape().to_vec();
            let src_dtype = arr.dtype();
            let mut result = RumpyArray::zeros(result_shape.clone(), src_dtype.clone());
            let src_ptr = arr.data_ptr();
            let src_ops = src_dtype.ops();
            let dst_dtype = result.dtype();
            let buffer = Arc::get_mut(result.buffer_mut()).expect("unique");
            let dst_ptr = buffer.as_mut_ptr();
            let dst_ops = dst_dtype.ops();

            let mut dst_indices = vec![0usize; ndim];
            let size = arr.size();

            for _ in 0..size {
                // Source index: roll along axis
                let mut src_indices = dst_indices.clone();
                src_indices[axis] = (dst_indices[axis] + axis_len - shift) % axis_len;

                let src_byte_offset = arr.byte_offset_for(&src_indices);
                let val = unsafe { src_ops.read_f64(src_ptr, src_byte_offset) }.unwrap_or(0.0);

                let dst_offset = linear_index(&dst_indices, &result_shape);
                unsafe { dst_ops.write_f64(dst_ptr, dst_offset, val); }

                increment_indices(&mut dst_indices, arr.shape());
            }

            result
        }
    }
}

/// Rotate array 90 degrees k times in the plane specified by axes.
/// Returns a view (no copy) like numpy.
pub fn rot90(arr: &RumpyArray, k: isize, axis0: usize, axis1: usize) -> RumpyArray {
    // Normalize k to 0-3
    let k = ((k % 4) + 4) as usize % 4;

    let ndim = arr.ndim();
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.swap(axis0, axis1);

    match k {
        0 => arr.clone(),
        1 => {
            // k=1: flip axis1, then transpose
            let flipped = arr.flip(axis1).unwrap();
            flipped.transpose_axes(&axes)
        }
        2 => {
            // k=2: flip both axes (180 degree rotation), no transpose
            let f1 = arr.flip(axis0).unwrap();
            f1.flip(axis1).unwrap()
        }
        3 => {
            // k=3 = k=-1: flip axis0, then transpose
            let flipped = arr.flip(axis0).unwrap();
            flipped.transpose_axes(&axes)
        }
        _ => unreachable!()
    }
}
