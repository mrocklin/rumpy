//! Set operations for arrays.
//!
//! Implements NumPy's set operations: isin, in1d, intersect1d, union1d, setdiff1d, setxor1d.
//!
//! Uses sorting-based algorithms (like NumPy) rather than hash-based for better performance
//! with sorted output requirements.

use crate::array::dtype::DTypeKind;
use crate::array::{DType, RumpyArray};
use std::collections::HashSet;
use std::sync::Arc;

/// Collect array values as f64 Vec (for any contiguous array).
fn collect_values(arr: &RumpyArray) -> Vec<f64> {
    if arr.is_c_contiguous() {
        let ptr = arr.data_ptr();
        let size = arr.size();
        match arr.dtype().kind() {
            DTypeKind::Float64 => {
                let p = ptr as *const f64;
                (0..size).map(|i| unsafe { *p.add(i) }).collect()
            }
            DTypeKind::Int64 => {
                let p = ptr as *const i64;
                (0..size).map(|i| unsafe { *p.add(i) as f64 }).collect()
            }
            DTypeKind::Int32 => {
                let p = ptr as *const i32;
                (0..size).map(|i| unsafe { *p.add(i) as f64 }).collect()
            }
            DTypeKind::Float32 => {
                let p = ptr as *const f32;
                (0..size).map(|i| unsafe { *p.add(i) as f64 }).collect()
            }
            _ => {
                let dtype = arr.dtype();
                let ops = dtype.ops();
                arr.iter_offsets()
                    .map(|offset| unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0))
                    .collect()
            }
        }
    } else {
        let dtype = arr.dtype();
        let ops = dtype.ops();
        let ptr = arr.data_ptr();
        arr.iter_offsets()
            .map(|offset| unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0))
            .collect()
    }
}

/// Extract values from array as sorted unique f64 Vec.
fn collect_sorted_unique(arr: &RumpyArray) -> Vec<f64> {
    let mut values = collect_values(arr);
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup();
    values
}

/// Build a HashSet from array values for O(1) lookup.
fn collect_hashset(arr: &RumpyArray) -> HashSet<u64> {
    collect_values(arr).into_iter().map(|v| v.to_bits()).collect()
}

/// Test whether each element of `element` is also present in `test_elements`.
/// Returns a boolean array of the same shape as `element`.
///
/// If `invert` is true, returns the inverse (elements NOT in test_elements).
pub fn isin(element: &RumpyArray, test_elements: &RumpyArray, invert: bool) -> RumpyArray {
    let test_set = collect_hashset(test_elements);
    let values = collect_values(element);

    // Create result array with same shape as element
    let mut result = RumpyArray::zeros(element.shape().to_vec(), DType::bool());

    if result.size() == 0 {
        return result;
    }

    let buffer = result.buffer_mut();
    let result_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let result_ptr = result_buffer.as_mut_ptr();

    for (i, val) in values.iter().enumerate() {
        let found = test_set.contains(&val.to_bits());
        unsafe { *result_ptr.add(i) = if found != invert { 1u8 } else { 0u8 }; }
    }

    result
}

/// Test whether each element of a 1D array is also present in a second array.
/// Like isin but always flattens the first argument.
pub fn in1d(ar1: &RumpyArray, ar2: &RumpyArray, invert: bool) -> RumpyArray {
    let flat = ar1.reshape(vec![ar1.size()]).unwrap_or_else(|| ar1.copy());
    isin(&flat, ar2, invert)
}

/// Find the intersection of two arrays using sorted merge.
/// Returns the sorted, unique values that are in both arrays.
pub fn intersect1d(ar1: &RumpyArray, ar2: &RumpyArray) -> RumpyArray {
    let v1 = collect_sorted_unique(ar1);
    let v2 = collect_sorted_unique(ar2);

    // Merge intersection on sorted arrays
    let mut result = Vec::with_capacity(v1.len().min(v2.len()));
    let (mut i, mut j) = (0, 0);
    while i < v1.len() && j < v2.len() {
        if v1[i] < v2[j] {
            i += 1;
        } else if v1[i] > v2[j] {
            j += 1;
        } else {
            result.push(v1[i]);
            i += 1;
            j += 1;
        }
    }

    let dtype = crate::array::promote_dtype(&ar1.dtype(), &ar2.dtype());
    RumpyArray::from_vec(result, dtype)
}

/// Find the union of two arrays using sorted merge.
/// Returns the sorted, unique values that are in either array.
pub fn union1d(ar1: &RumpyArray, ar2: &RumpyArray) -> RumpyArray {
    let v1 = collect_sorted_unique(ar1);
    let v2 = collect_sorted_unique(ar2);

    // Merge union on sorted arrays
    let mut result = Vec::with_capacity(v1.len() + v2.len());
    let (mut i, mut j) = (0, 0);
    while i < v1.len() && j < v2.len() {
        if v1[i] < v2[j] {
            result.push(v1[i]);
            i += 1;
        } else if v1[i] > v2[j] {
            result.push(v2[j]);
            j += 1;
        } else {
            result.push(v1[i]);
            i += 1;
            j += 1;
        }
    }
    result.extend_from_slice(&v1[i..]);
    result.extend_from_slice(&v2[j..]);

    let dtype = crate::array::promote_dtype(&ar1.dtype(), &ar2.dtype());
    RumpyArray::from_vec(result, dtype)
}

/// Find the set difference of two arrays using sorted merge.
/// Returns the sorted, unique values in ar1 that are not in ar2.
pub fn setdiff1d(ar1: &RumpyArray, ar2: &RumpyArray) -> RumpyArray {
    let v1 = collect_sorted_unique(ar1);
    let v2 = collect_sorted_unique(ar2);

    // Merge difference on sorted arrays (elements in v1 not in v2)
    let mut result = Vec::with_capacity(v1.len());
    let (mut i, mut j) = (0, 0);
    while i < v1.len() {
        if j >= v2.len() || v1[i] < v2[j] {
            result.push(v1[i]);
            i += 1;
        } else if v1[i] > v2[j] {
            j += 1;
        } else {
            // Equal - skip this element from v1
            i += 1;
            j += 1;
        }
    }

    RumpyArray::from_vec(result, ar1.dtype())
}

/// Find the symmetric difference of two arrays using sorted merge.
/// Returns the sorted, unique values that are in exactly one of the arrays.
pub fn setxor1d(ar1: &RumpyArray, ar2: &RumpyArray) -> RumpyArray {
    let v1 = collect_sorted_unique(ar1);
    let v2 = collect_sorted_unique(ar2);

    // Merge symmetric difference on sorted arrays
    let mut result = Vec::with_capacity(v1.len() + v2.len());
    let (mut i, mut j) = (0, 0);
    while i < v1.len() && j < v2.len() {
        if v1[i] < v2[j] {
            result.push(v1[i]);
            i += 1;
        } else if v1[i] > v2[j] {
            result.push(v2[j]);
            j += 1;
        } else {
            // Equal - skip both
            i += 1;
            j += 1;
        }
    }
    result.extend_from_slice(&v1[i..]);
    result.extend_from_slice(&v2[j..]);

    let dtype = crate::array::promote_dtype(&ar1.dtype(), &ar2.dtype());
    RumpyArray::from_vec(result, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isin_basic() {
        let element = RumpyArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::float64());
        let test = RumpyArray::from_vec(vec![2.0, 4.0], DType::float64());
        let result = isin(&element, &test, false);
        assert_eq!(result.size(), 5);
        // Values at indices 1 and 3 should be true (1.0)
    }

    #[test]
    fn test_intersect1d_basic() {
        let a = RumpyArray::from_vec(vec![1.0, 2.0, 3.0], DType::float64());
        let b = RumpyArray::from_vec(vec![2.0, 3.0, 4.0], DType::float64());
        let result = intersect1d(&a, &b);
        assert_eq!(result.size(), 2);
    }

    #[test]
    fn test_union1d_basic() {
        let a = RumpyArray::from_vec(vec![1.0, 2.0, 3.0], DType::float64());
        let b = RumpyArray::from_vec(vec![3.0, 4.0, 5.0], DType::float64());
        let result = union1d(&a, &b);
        assert_eq!(result.size(), 5);
    }

    #[test]
    fn test_setdiff1d_basic() {
        let a = RumpyArray::from_vec(vec![1.0, 2.0, 3.0], DType::float64());
        let b = RumpyArray::from_vec(vec![2.0], DType::float64());
        let result = setdiff1d(&a, &b);
        assert_eq!(result.size(), 2);
    }

    #[test]
    fn test_setxor1d_basic() {
        let a = RumpyArray::from_vec(vec![1.0, 2.0, 3.0], DType::float64());
        let b = RumpyArray::from_vec(vec![2.0, 3.0, 4.0], DType::float64());
        let result = setxor1d(&a, &b);
        assert_eq!(result.size(), 2); // 1 and 4
    }
}
