//! Logical operations on RumpyArray (all, any, count_nonzero).

use crate::array::{increment_indices, DType, RumpyArray};
use std::sync::Arc;

impl RumpyArray {
    /// Count number of non-zero elements.
    pub fn count_nonzero(&self) -> usize {
        let size = self.size();
        if size == 0 {
            return 0;
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        let mut count = 0;
        for offset in self.iter_offsets() {
            if let Some(val) = unsafe { ops.read_f64(ptr, offset) } {
                if val != 0.0 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Test if all elements evaluate to True.
    pub fn all(&self) -> bool {
        let size = self.size();
        if size == 0 {
            return true; // numpy convention: empty array is all True
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        for offset in self.iter_offsets() {
            if let Some(val) = unsafe { ops.read_f64(ptr, offset) } {
                if val == 0.0 {
                    return false;
                }
            }
        }
        true
    }

    /// Test if all elements along axis evaluate to True.
    pub fn all_axis(&self, axis: usize) -> RumpyArray {
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        let axis_len = out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let mut result = RumpyArray::zeros(out_shape.clone(), DType::bool());
        let out_size = result.size();
        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let bool_dtype = DType::bool();
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = bool_dtype.ops();

        let mut out_indices = vec![0usize; out_shape.len()];
        for i in 0..out_size {
            let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 {
                in_indices.extend_from_slice(&out_indices[axis..]);
            }

            let mut result_val = true;
            for j in 0..axis_len {
                in_indices[axis] = j;
                if self.get_element(&in_indices) == 0.0 {
                    result_val = false;
                    break;
                }
            }

            unsafe { ops.write_f64(result_ptr, i, if result_val { 1.0 } else { 0.0 }); }
            increment_indices(&mut out_indices, &out_shape);
        }

        result
    }

    /// Test if any element evaluates to True.
    pub fn any(&self) -> bool {
        let size = self.size();
        if size == 0 {
            return false; // numpy convention: empty array is all False
        }
        let ptr = self.data_ptr();
        let dtype = self.dtype();
        let ops = dtype.ops();
        for offset in self.iter_offsets() {
            if let Some(val) = unsafe { ops.read_f64(ptr, offset) } {
                if val != 0.0 {
                    return true;
                }
            }
        }
        false
    }

    /// Test if any element along axis evaluates to True.
    pub fn any_axis(&self, axis: usize) -> RumpyArray {
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        let axis_len = out_shape.remove(axis);
        if out_shape.is_empty() {
            out_shape = vec![1];
        }

        let mut result = RumpyArray::zeros(out_shape.clone(), DType::bool());
        let out_size = result.size();
        if out_size == 0 || axis_len == 0 {
            return result;
        }

        let bool_dtype = DType::bool();
        let buffer = result.buffer_mut();
        let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
        let result_ptr = result_buffer.as_mut_ptr();
        let ops = bool_dtype.ops();

        let mut out_indices = vec![0usize; out_shape.len()];
        for i in 0..out_size {
            let mut in_indices: Vec<usize> = out_indices[..axis.min(out_indices.len())].to_vec();
            in_indices.push(0);
            if axis < self.ndim() - 1 {
                in_indices.extend_from_slice(&out_indices[axis..]);
            }

            let mut result_val = false;
            for j in 0..axis_len {
                in_indices[axis] = j;
                if self.get_element(&in_indices) != 0.0 {
                    result_val = true;
                    break;
                }
            }

            unsafe { ops.write_f64(result_ptr, i, if result_val { 1.0 } else { 0.0 }); }
            increment_indices(&mut out_indices, &out_shape);
        }

        result
    }
}
