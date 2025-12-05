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
            unsafe { ops.write_f64(ptr, i, v); }
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

        // If same dtype, just copy elements
        if self.dtype == new_dtype {
            let mut indices = vec![0usize; self.ndim()];
            for i in 0..size {
                let src_offset = self.byte_offset_for(&indices);
                unsafe { dst_ops.copy_element(src_ptr, src_offset, result_ptr, i); }
                increment_indices(&mut indices, self.shape());
            }
        } else {
            // Cross-dtype conversion via f64 (lossy for complex)
            let mut indices = vec![0usize; self.ndim()];
            for i in 0..size {
                let val = self.get_element(&indices);
                unsafe { dst_ops.write_f64(result_ptr, i, val); }
                increment_indices(&mut indices, self.shape());
            }
        }
        result
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
        let needs_dtype = match self.dtype.kind() {
            DTypeKind::Int64 | DTypeKind::Float64 | DTypeKind::Bool => false,
            _ => true,
        };

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
    if axis_len % num_sections != 0 {
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
fn split_into_sizes(arr: &RumpyArray, sizes: &[usize], axis: usize) -> Vec<RumpyArray> {
    let mut result = Vec::with_capacity(sizes.len());
    let mut start = 0isize;

    for &size in sizes {
        let section = arr.slice_axis(axis, start, start + size as isize, 1);
        // Make contiguous copy
        result.push(section.copy());
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
