//! View operations: reshape, transpose, flip, slice_axis, broadcast_to, squeeze, expand_dims.

use super::flags::ArrayFlags;
use super::{compute_c_strides, compute_f_strides, is_c_contiguous, is_f_contiguous, RumpyArray};
use std::sync::Arc;

impl RumpyArray {
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
    pub fn view_with_dtype(
        &self,
        shape: Vec<usize>,
        strides: Vec<isize>,
        dtype: super::dtype::DType,
    ) -> Self {
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
            if stop > start {
                (stop - start + step - 1) / step
            } else {
                0
            }
        } else if start > stop {
            (start - stop - step - 1) / (-step)
        } else {
            0
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

    /// Transpose with specified axis order. Returns a view.
    pub fn transpose_axes(&self, axes: &[usize]) -> Self {
        assert_eq!(
            axes.len(),
            self.ndim(),
            "axes must have same length as ndim"
        );
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

    /// Remove single-dimensional entries from the shape.
    /// Returns a view.
    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape().iter().copied().filter(|&d| d != 1).collect();
        let new_strides: Vec<isize> = self
            .shape()
            .iter()
            .zip(self.strides())
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
            return self.view_with(
                offset_delta.max(0) as usize,
                vec![1],
                vec![self.itemsize() as isize],
            );
        }

        self.view_with(offset_delta.max(0) as usize, core_shape, core_strides)
    }
}
