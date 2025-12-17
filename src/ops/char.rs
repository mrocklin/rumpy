//! Vectorized string operations (numpy.char module equivalent).
//!
//! These operations work element-wise on string arrays.

use crate::array::dtype::{DType, DTypeKind};
use crate::array::RumpyArray;
use std::sync::Arc;

// ============================================================================
// Direct UTF-32 operations (fast path, no String allocation)
// ============================================================================

/// Convert a single UTF-32 code point to uppercase.
/// Returns the uppercase code point, or the original if no mapping exists.
/// Note: This only handles 1:1 mappings. Characters like 'ß' that expand
/// to multiple characters ("SS") will not be converted.
#[inline(always)]
fn codepoint_to_upper(cp: u32) -> u32 {
    // Fast path: ASCII lowercase
    if cp >= 0x61 && cp <= 0x7A {
        return cp - 32;
    }
    // Non-ASCII: use char conversion (1:1 mappings only)
    if let Some(c) = char::from_u32(cp) {
        let mut upper = c.to_uppercase();
        if let Some(uc) = upper.next() {
            // Only use if it's a 1:1 mapping
            if upper.next().is_none() {
                return uc as u32;
            }
        }
    }
    cp
}

/// Convert a single UTF-32 code point to lowercase.
#[inline(always)]
fn codepoint_to_lower(cp: u32) -> u32 {
    // Fast path: ASCII uppercase
    if cp >= 0x41 && cp <= 0x5A {
        return cp + 32;
    }
    // Non-ASCII: use char conversion (1:1 mappings only)
    if let Some(c) = char::from_u32(cp) {
        let mut lower = c.to_lowercase();
        if let Some(lc) = lower.next() {
            if lower.next().is_none() {
                return lc as u32;
            }
        }
    }
    cp
}

/// Apply a per-codepoint transformation on a string array.
/// Handles both contiguous and strided arrays correctly.
fn apply_codepoint_transform_strided<F>(arr: &RumpyArray, transform: F) -> Option<RumpyArray>
where
    F: Fn(u32) -> u32,
{
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let size = arr.size();
    let dtype = arr.dtype().clone();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), dtype);

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst_base = out_buffer.as_mut_ptr() as *mut u32;

    if arr.is_c_contiguous() {
        // Fast path: contiguous data - process all code points directly
        let src_ptr = arr.data_ptr() as *const u32;
        let total = size * max_chars;

        // Process in chunks of 8 for SIMD-friendly auto-vectorization
        let chunks = total / 8;
        let remainder = total % 8;

        unsafe {
            for chunk in 0..chunks {
                let base = chunk * 8;
                for j in 0..8 {
                    let cp = *src_ptr.add(base + j);
                    *dst_base.add(base + j) = if cp == 0 { 0 } else { transform(cp) };
                }
            }
            let base = chunks * 8;
            for j in 0..remainder {
                let cp = *src_ptr.add(base + j);
                *dst_base.add(base + j) = if cp == 0 { 0 } else { transform(cp) };
            }
        }
    } else {
        // Slow path: strided data - use iter_offsets for correct element access
        let src_base = arr.data_ptr();

        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let dst_str = unsafe { dst_base.add(elem_idx * max_chars) };

            for char_idx in 0..max_chars {
                unsafe {
                    let cp = *src_str.add(char_idx);
                    *dst_str.add(char_idx) = if cp == 0 { 0 } else { transform(cp) };
                }
            }
        }
    }

    Some(out)
}

/// Fast uppercase transformation with ASCII optimization.
fn apply_upper_fast(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_codepoint_transform_strided(arr, |cp| {
        // Fast path: ASCII lowercase
        if cp >= 0x61 && cp <= 0x7A {
            cp - 32
        } else if cp <= 0x7F {
            cp  // ASCII non-lowercase stays as-is
        } else {
            codepoint_to_upper(cp)
        }
    })
}

/// Fast lowercase transformation with ASCII optimization.
fn apply_lower_fast(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_codepoint_transform_strided(arr, |cp| {
        // Fast path: ASCII uppercase
        if cp >= 0x41 && cp <= 0x5A {
            cp + 32
        } else if cp <= 0x7F {
            cp  // ASCII non-uppercase stays as-is
        } else {
            codepoint_to_lower(cp)
        }
    })
}

// ============================================================================
// Direct UTF-32 predicate operations
// ============================================================================

/// Get the actual length (non-null code points) of a UTF-32 string.
#[inline]
unsafe fn utf32_strlen(ptr: *const u32, max_chars: usize) -> usize {
    for i in 0..max_chars {
        if *ptr.add(i) == 0 {
            return i;
        }
    }
    max_chars
}

/// Apply a predicate to each string element, returning a bool array.
/// The predicate receives a slice of code points (excluding trailing nulls).
fn apply_predicate_fast<F>(arr: &RumpyArray, predicate: F) -> Option<RumpyArray>
where
    F: Fn(&[u32]) -> bool,
{
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let size = arr.size();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), DType::bool());

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst = out_buffer.as_mut_ptr();

    // Temporary buffer for code points (reused across elements)
    let mut codepoints = vec![0u32; max_chars];

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            // Copy to buffer
            for i in 0..len {
                codepoints[i] = unsafe { *src_str.add(i) };
            }

            let result = predicate(&codepoints[..len]);
            unsafe { *dst.add(elem_idx) = if result { 1 } else { 0 } };
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            for i in 0..len {
                codepoints[i] = unsafe { *src_str.add(i) };
            }

            let result = predicate(&codepoints[..len]);
            unsafe { *dst.add(elem_idx) = if result { 1 } else { 0 } };
        }
    }

    Some(out)
}

// ============================================================================
// Code point predicates
// ============================================================================

#[inline]
fn cp_is_alpha(cp: u32) -> bool {
    if cp <= 0x7F {
        (cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A)
    } else {
        char::from_u32(cp).map_or(false, |c| c.is_alphabetic())
    }
}

#[inline]
fn cp_is_digit(cp: u32) -> bool {
    if cp <= 0x7F {
        cp >= 0x30 && cp <= 0x39
    } else {
        char::from_u32(cp).map_or(false, |c| c.is_ascii_digit())
    }
}

#[inline]
fn cp_is_alnum(cp: u32) -> bool {
    cp_is_alpha(cp) || cp_is_digit(cp)
}

#[inline]
fn cp_is_space(cp: u32) -> bool {
    if cp <= 0x7F {
        cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D
    } else {
        char::from_u32(cp).map_or(false, |c| c.is_whitespace())
    }
}

#[inline]
fn cp_is_upper(cp: u32) -> bool {
    if cp <= 0x7F {
        cp >= 0x41 && cp <= 0x5A
    } else {
        char::from_u32(cp).map_or(false, |c| c.is_uppercase())
    }
}

#[inline]
fn cp_is_lower(cp: u32) -> bool {
    if cp <= 0x7F {
        cp >= 0x61 && cp <= 0x7A
    } else {
        char::from_u32(cp).map_or(false, |c| c.is_lowercase())
    }
}

#[inline]
fn cp_is_cased(cp: u32) -> bool {
    cp_is_upper(cp) || cp_is_lower(cp)
}

/// Read a string from a string array at the given flat index.
///
/// # Safety
/// The array must have a string dtype and the index must be valid.
unsafe fn read_string(arr: &RumpyArray, flat_idx: usize) -> String {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return String::new(),
    };

    let itemsize = arr.dtype().itemsize();
    let ptr = arr.data_ptr();
    let byte_offset = (flat_idx * itemsize) as isize;
    let start = ptr.offset(byte_offset) as *const u32;

    let mut chars = Vec::with_capacity(max_chars);
    for i in 0..max_chars {
        let code_point = *start.add(i);
        if code_point == 0 {
            break;
        }
        if let Some(c) = char::from_u32(code_point) {
            chars.push(c);
        }
    }
    chars.into_iter().collect()
}

/// Write a string to a string array at the given flat index.
///
/// # Safety
/// The array must have a string dtype, be writable, and the index must be valid.
unsafe fn write_string(arr: &mut RumpyArray, flat_idx: usize, s: &str) {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return,
    };

    let itemsize = arr.dtype().itemsize();
    let buffer = Arc::get_mut(&mut arr.buffer).expect("buffer must be unique");
    let ptr = buffer.as_mut_ptr().add(arr.offset);
    let byte_offset = (flat_idx * itemsize) as isize;
    let start = ptr.offset(byte_offset) as *mut u32;

    let mut written = 0;
    for c in s.chars().take(max_chars) {
        *start.add(written) = c as u32;
        written += 1;
    }
    // Pad with nulls
    for i in written..max_chars {
        *start.add(i) = 0;
    }
}

/// Apply a unary string operation element-wise.
fn apply_unary_string_op<F>(arr: &RumpyArray, op: F) -> Option<RumpyArray>
where
    F: Fn(&str) -> String,
{
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let size = arr.size();

    // Compute max output length
    let mut max_out_len = 0usize;
    let mut results = Vec::with_capacity(size);
    for i in 0..size {
        let s = unsafe { read_string(arr, i) };
        let result = op(&s);
        max_out_len = max_out_len.max(result.chars().count());
        results.push(result);
    }

    // Use same max_chars as input if output fits, otherwise expand
    let out_max_chars = max_chars.max(max_out_len);
    let out_dtype = DType::str_(out_max_chars);
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), out_dtype);

    for (i, result) in results.into_iter().enumerate() {
        unsafe { write_string(&mut out, i, &result) };
    }

    Some(out)
}

/// Apply a binary string operation element-wise with broadcasting.
fn apply_binary_string_op<F>(a: &RumpyArray, b: &RumpyArray, op: F) -> Option<RumpyArray>
where
    F: Fn(&str, &str) -> String,
{
    let a_max = match a.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };
    let b_max = match b.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    // For now, require same shape (no broadcasting)
    if a.shape() != b.shape() {
        return None;
    }

    let size = a.size();
    let mut max_out_len = 0usize;
    let mut results = Vec::with_capacity(size);

    for i in 0..size {
        let sa = unsafe { read_string(a, i) };
        let sb = unsafe { read_string(b, i) };
        let result = op(&sa, &sb);
        max_out_len = max_out_len.max(result.chars().count());
        results.push(result);
    }

    let out_max_chars = a_max.max(b_max).max(max_out_len);
    let out_dtype = DType::str_(out_max_chars);
    let mut out = RumpyArray::zeros(a.shape().to_vec(), out_dtype);

    for (i, result) in results.into_iter().enumerate() {
        unsafe { write_string(&mut out, i, &result) };
    }

    Some(out)
}

// ============================================================================
// Tier 1 - Essential Operations
// ============================================================================

/// Concatenate two string arrays element-wise.
pub fn add(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    apply_binary_string_op(a, b, |sa, sb| format!("{}{}", sa, sb))
}

/// Repeat each string n times.
pub fn multiply(arr: &RumpyArray, n: usize) -> Option<RumpyArray> {
    apply_unary_string_op(arr, |s| s.repeat(n))
}

/// Convert to uppercase.
pub fn upper(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_upper_fast(arr)
}

/// Convert to lowercase.
pub fn lower(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_lower_fast(arr)
}

/// Strip helper - processes directly without intermediate allocations.
/// mode: 0 = strip both, 1 = lstrip, 2 = rstrip
fn strip_fast(arr: &RumpyArray, mode: u8) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let size = arr.size();
    let dtype = arr.dtype().clone();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), dtype);

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst_base = out_buffer.as_mut_ptr() as *mut u32;

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let dst_str = unsafe { dst_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            // Find start (skip leading whitespace)
            let start = if mode == 2 { 0 } else {
                let mut s = 0;
                while s < len && cp_is_space(unsafe { *src_str.add(s) }) { s += 1; }
                s
            };

            // Find end (skip trailing whitespace)
            let end = if mode == 1 { len } else {
                let mut e = len;
                while e > start && cp_is_space(unsafe { *src_str.add(e - 1) }) { e -= 1; }
                e
            };

            // Copy the result
            let result_len = end - start;
            unsafe {
                std::ptr::copy_nonoverlapping(src_str.add(start), dst_str, result_len);
            }
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let dst_str = unsafe { dst_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let start = if mode == 2 { 0 } else {
                let mut s = 0;
                while s < len && cp_is_space(unsafe { *src_str.add(s) }) { s += 1; }
                s
            };

            let end = if mode == 1 { len } else {
                let mut e = len;
                while e > start && cp_is_space(unsafe { *src_str.add(e - 1) }) { e -= 1; }
                e
            };

            let result_len = end - start;
            unsafe {
                std::ptr::copy_nonoverlapping(src_str.add(start), dst_str, result_len);
            }
        }
    }

    Some(out)
}

/// Strip leading and trailing whitespace.
pub fn strip(arr: &RumpyArray) -> Option<RumpyArray> {
    strip_fast(arr, 0)
}

/// Strip leading whitespace.
pub fn lstrip(arr: &RumpyArray) -> Option<RumpyArray> {
    strip_fast(arr, 1)
}

/// Strip trailing whitespace.
pub fn rstrip(arr: &RumpyArray) -> Option<RumpyArray> {
    strip_fast(arr, 2)
}

// ============================================================================
// Tier 2 - Search/Replace Operations
// ============================================================================

/// Find substring - optimized direct implementation.
pub fn find(arr: &RumpyArray, sub: &str) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let sub_cps: Vec<u32> = sub.chars().map(|c| c as u32).collect();
    let sub_len = sub_cps.len();

    let size = arr.size();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), DType::int64());

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst = out_buffer.as_mut_ptr() as *mut i64;

    // Empty substring returns 0
    if sub_len == 0 {
        // All zeros already from RumpyArray::zeros
        return Some(out);
    }

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            // Search for substring directly in memory
            let result = if len < sub_len {
                -1
            } else {
                let mut found: i64 = -1;
                'outer: for i in 0..=(len - sub_len) {
                    for j in 0..sub_len {
                        if unsafe { *src_str.add(i + j) } != sub_cps[j] {
                            continue 'outer;
                        }
                    }
                    found = i as i64;
                    break;
                }
                found
            };
            unsafe { *dst.add(elem_idx) = result };
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let result = if len < sub_len {
                -1
            } else {
                let mut found: i64 = -1;
                'outer: for i in 0..=(len - sub_len) {
                    for j in 0..sub_len {
                        if unsafe { *src_str.add(i + j) } != sub_cps[j] {
                            continue 'outer;
                        }
                    }
                    found = i as i64;
                    break;
                }
                found
            };
            unsafe { *dst.add(elem_idx) = result };
        }
    }

    Some(out)
}

/// Find the highest index of substring in each string.
/// Returns -1 if not found.
pub fn rfind(arr: &RumpyArray, sub: &str) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let sub_cps: Vec<u32> = sub.chars().map(|c| c as u32).collect();
    let sub_len = sub_cps.len();

    let size = arr.size();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), DType::int64());

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst = out_buffer.as_mut_ptr() as *mut i64;

    if sub_len == 0 {
        // Empty substring: return len of each string
        if arr.is_c_contiguous() {
            let src_base = arr.data_ptr() as *const u32;
            for elem_idx in 0..size {
                let src_str = unsafe { src_base.add(elem_idx * max_chars) };
                let len = unsafe { utf32_strlen(src_str, max_chars) };
                unsafe { *dst.add(elem_idx) = len as i64 };
            }
        } else {
            let src_base = arr.data_ptr();
            for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
                let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
                let len = unsafe { utf32_strlen(src_str, max_chars) };
                unsafe { *dst.add(elem_idx) = len as i64 };
            }
        }
        return Some(out);
    }

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            // Search backwards
            let result = if len < sub_len {
                -1
            } else {
                let mut found: i64 = -1;
                for i in (0..=(len - sub_len)).rev() {
                    let mut matches = true;
                    for j in 0..sub_len {
                        if unsafe { *src_str.add(i + j) } != sub_cps[j] {
                            matches = false;
                            break;
                        }
                    }
                    if matches {
                        found = i as i64;
                        break;
                    }
                }
                found
            };
            unsafe { *dst.add(elem_idx) = result };
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let result = if len < sub_len {
                -1
            } else {
                let mut found: i64 = -1;
                for i in (0..=(len - sub_len)).rev() {
                    let mut matches = true;
                    for j in 0..sub_len {
                        if unsafe { *src_str.add(i + j) } != sub_cps[j] {
                            matches = false;
                            break;
                        }
                    }
                    if matches {
                        found = i as i64;
                        break;
                    }
                }
                found
            };
            unsafe { *dst.add(elem_idx) = result };
        }
    }

    Some(out)
}

/// Like find, but returns -1 sentinel for "not found" (Python wrapper raises ValueError).
pub fn index(arr: &RumpyArray, sub: &str) -> Option<RumpyArray> {
    // Same as find - Python wrapper checks for -1 and raises ValueError
    find(arr, sub)
}

/// Like rfind, but returns -1 sentinel for "not found" (Python wrapper raises ValueError).
pub fn rindex(arr: &RumpyArray, sub: &str) -> Option<RumpyArray> {
    // Same as rfind - Python wrapper checks for -1 and raises ValueError
    rfind(arr, sub)
}

/// Partition each string at first occurrence of separator.
/// Returns array with shape (*input_shape, 3) containing [before, sep, after].
pub fn partition(arr: &RumpyArray, sep: &str) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let sep_cps: Vec<u32> = sep.chars().map(|c| c as u32).collect();
    let sep_len = sep_cps.len();

    if sep_len == 0 {
        return None; // Empty separator not allowed
    }

    let size = arr.size();
    let mut out_shape = arr.shape().to_vec();
    out_shape.push(3);

    let out_dtype = DType::str_(max_chars);
    let mut out = RumpyArray::zeros(out_shape, out_dtype);

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst_base = out_buffer.as_mut_ptr() as *mut u32;

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            // Find separator
            let mut found_at: Option<usize> = None;
            if len >= sep_len {
                'outer: for i in 0..=(len - sep_len) {
                    for j in 0..sep_len {
                        if unsafe { *src_str.add(i + j) } != sep_cps[j] {
                            continue 'outer;
                        }
                    }
                    found_at = Some(i);
                    break;
                }
            }

            let dst_before = unsafe { dst_base.add((elem_idx * 3) * max_chars) };
            let dst_sep = unsafe { dst_base.add((elem_idx * 3 + 1) * max_chars) };
            let dst_after = unsafe { dst_base.add((elem_idx * 3 + 2) * max_chars) };

            match found_at {
                Some(idx) => {
                    // Copy before
                    for i in 0..idx {
                        unsafe { *dst_before.add(i) = *src_str.add(i); }
                    }
                    // Copy separator
                    for i in 0..sep_len {
                        unsafe { *dst_sep.add(i) = sep_cps[i]; }
                    }
                    // Copy after
                    let after_start = idx + sep_len;
                    for i in after_start..len {
                        unsafe { *dst_after.add(i - after_start) = *src_str.add(i); }
                    }
                }
                None => {
                    // No separator found: (string, "", "")
                    for i in 0..len {
                        unsafe { *dst_before.add(i) = *src_str.add(i); }
                    }
                }
            }
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let mut found_at: Option<usize> = None;
            if len >= sep_len {
                'outer: for i in 0..=(len - sep_len) {
                    for j in 0..sep_len {
                        if unsafe { *src_str.add(i + j) } != sep_cps[j] {
                            continue 'outer;
                        }
                    }
                    found_at = Some(i);
                    break;
                }
            }

            let dst_before = unsafe { dst_base.add((elem_idx * 3) * max_chars) };
            let dst_sep = unsafe { dst_base.add((elem_idx * 3 + 1) * max_chars) };
            let dst_after = unsafe { dst_base.add((elem_idx * 3 + 2) * max_chars) };

            match found_at {
                Some(idx) => {
                    for i in 0..idx {
                        unsafe { *dst_before.add(i) = *src_str.add(i); }
                    }
                    for i in 0..sep_len {
                        unsafe { *dst_sep.add(i) = sep_cps[i]; }
                    }
                    let after_start = idx + sep_len;
                    for i in after_start..len {
                        unsafe { *dst_after.add(i - after_start) = *src_str.add(i); }
                    }
                }
                None => {
                    for i in 0..len {
                        unsafe { *dst_before.add(i) = *src_str.add(i); }
                    }
                }
            }
        }
    }

    Some(out)
}

/// Partition each string at last occurrence of separator.
/// Returns array with shape (*input_shape, 3) containing [before, sep, after].
pub fn rpartition(arr: &RumpyArray, sep: &str) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let sep_cps: Vec<u32> = sep.chars().map(|c| c as u32).collect();
    let sep_len = sep_cps.len();

    if sep_len == 0 {
        return None;
    }

    let size = arr.size();
    let mut out_shape = arr.shape().to_vec();
    out_shape.push(3);

    let out_dtype = DType::str_(max_chars);
    let mut out = RumpyArray::zeros(out_shape, out_dtype);

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst_base = out_buffer.as_mut_ptr() as *mut u32;

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            // Find last separator
            let mut found_at: Option<usize> = None;
            if len >= sep_len {
                for i in (0..=(len - sep_len)).rev() {
                    let mut matches = true;
                    for j in 0..sep_len {
                        if unsafe { *src_str.add(i + j) } != sep_cps[j] {
                            matches = false;
                            break;
                        }
                    }
                    if matches {
                        found_at = Some(i);
                        break;
                    }
                }
            }

            let dst_before = unsafe { dst_base.add((elem_idx * 3) * max_chars) };
            let dst_sep = unsafe { dst_base.add((elem_idx * 3 + 1) * max_chars) };
            let dst_after = unsafe { dst_base.add((elem_idx * 3 + 2) * max_chars) };

            match found_at {
                Some(idx) => {
                    for i in 0..idx {
                        unsafe { *dst_before.add(i) = *src_str.add(i); }
                    }
                    for i in 0..sep_len {
                        unsafe { *dst_sep.add(i) = sep_cps[i]; }
                    }
                    let after_start = idx + sep_len;
                    for i in after_start..len {
                        unsafe { *dst_after.add(i - after_start) = *src_str.add(i); }
                    }
                }
                None => {
                    // No separator found: ("", "", string)
                    for i in 0..len {
                        unsafe { *dst_after.add(i) = *src_str.add(i); }
                    }
                }
            }
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let mut found_at: Option<usize> = None;
            if len >= sep_len {
                for i in (0..=(len - sep_len)).rev() {
                    let mut matches = true;
                    for j in 0..sep_len {
                        if unsafe { *src_str.add(i + j) } != sep_cps[j] {
                            matches = false;
                            break;
                        }
                    }
                    if matches {
                        found_at = Some(i);
                        break;
                    }
                }
            }

            let dst_before = unsafe { dst_base.add((elem_idx * 3) * max_chars) };
            let dst_sep = unsafe { dst_base.add((elem_idx * 3 + 1) * max_chars) };
            let dst_after = unsafe { dst_base.add((elem_idx * 3 + 2) * max_chars) };

            match found_at {
                Some(idx) => {
                    for i in 0..idx {
                        unsafe { *dst_before.add(i) = *src_str.add(i); }
                    }
                    for i in 0..sep_len {
                        unsafe { *dst_sep.add(i) = sep_cps[i]; }
                    }
                    let after_start = idx + sep_len;
                    for i in after_start..len {
                        unsafe { *dst_after.add(i - after_start) = *src_str.add(i); }
                    }
                }
                None => {
                    for i in 0..len {
                        unsafe { *dst_after.add(i) = *src_str.add(i); }
                    }
                }
            }
        }
    }

    Some(out)
}

/// Join: insert separator between each character of each string.
/// E.g., join('-', ['ab', 'cd']) -> ['a-b', 'c-d']
pub fn join(sep: &str, arr: &RumpyArray) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let sep_cps: Vec<u32> = sep.chars().map(|c| c as u32).collect();
    let sep_len = sep_cps.len();

    let size = arr.size();
    if size == 0 {
        return Some(RumpyArray::zeros(arr.shape().to_vec(), arr.dtype().clone()));
    }

    // Compute max output length: for string of length n, result is n + (n-1)*sep_len
    // max_out = max_chars + (max_chars - 1) * sep_len = max_chars * (1 + sep_len) - sep_len
    let max_out_chars = if max_chars == 0 { 0 } else { max_chars + (max_chars - 1) * sep_len };
    let out_dtype = DType::str_(max_out_chars);
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), out_dtype);

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst_base = out_buffer.as_mut_ptr() as *mut u32;

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let dst_str = unsafe { dst_base.add(elem_idx * max_out_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            if len == 0 {
                continue;
            }

            let mut dst_idx = 0;
            // First character
            unsafe { *dst_str.add(dst_idx) = *src_str; }
            dst_idx += 1;

            // Remaining characters with separator
            for i in 1..len {
                for &cp in &sep_cps {
                    unsafe { *dst_str.add(dst_idx) = cp; }
                    dst_idx += 1;
                }
                unsafe { *dst_str.add(dst_idx) = *src_str.add(i); }
                dst_idx += 1;
            }
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let dst_str = unsafe { dst_base.add(elem_idx * max_out_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            if len == 0 {
                continue;
            }

            let mut dst_idx = 0;
            unsafe { *dst_str.add(dst_idx) = *src_str; }
            dst_idx += 1;

            for i in 1..len {
                for &cp in &sep_cps {
                    unsafe { *dst_str.add(dst_idx) = cp; }
                    dst_idx += 1;
                }
                unsafe { *dst_str.add(dst_idx) = *src_str.add(i); }
                dst_idx += 1;
            }
        }
    }

    Some(out)
}

/// Replace occurrences of old with new in each string.
pub fn replace(arr: &RumpyArray, old: &str, new: &str, count: Option<usize>) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let old_cps: Vec<u32> = old.chars().map(|c| c as u32).collect();
    let new_cps: Vec<u32> = new.chars().map(|c| c as u32).collect();
    let old_len = old_cps.len();
    let new_len = new_cps.len();
    let max_count = count.unwrap_or(usize::MAX);

    let size = arr.size();

    if size == 0 {
        return Some(RumpyArray::zeros(arr.shape().to_vec(), arr.dtype().clone()));
    }

    // Empty old string: no replacements
    if old_len == 0 {
        return Some(arr.copy());
    }

    // First pass: compute max output length
    let mut max_out_len = 0usize;

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            // Count occurrences up to max_count
            let mut cnt = 0usize;
            let mut i = 0;
            while i + old_len <= len && cnt < max_count {
                let mut matches = true;
                for j in 0..old_len {
                    if unsafe { *src_str.add(i + j) } != old_cps[j] {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    cnt += 1;
                    i += old_len;
                } else {
                    i += 1;
                }
            }

            // Compute output length: original - (cnt * old_len) + (cnt * new_len)
            let out_len = len - cnt * old_len + cnt * new_len;
            max_out_len = max_out_len.max(out_len);
        }
    } else {
        let src_base = arr.data_ptr();
        for (_elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let mut cnt = 0usize;
            let mut i = 0;
            while i + old_len <= len && cnt < max_count {
                let mut matches = true;
                for j in 0..old_len {
                    if unsafe { *src_str.add(i + j) } != old_cps[j] {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    cnt += 1;
                    i += old_len;
                } else {
                    i += 1;
                }
            }
            let out_len = len - cnt * old_len + cnt * new_len;
            max_out_len = max_out_len.max(out_len);
        }
    }

    // Create output with appropriate size
    let out_max_chars = max_chars.max(max_out_len);
    let out_dtype = DType::str_(out_max_chars);
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), out_dtype);

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst_base = out_buffer.as_mut_ptr() as *mut u32;

    // Second pass: perform replacements
    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let dst_str = unsafe { dst_base.add(elem_idx * out_max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let mut src_i = 0;
            let mut dst_i = 0;
            let mut cnt = 0usize;

            while src_i < len {
                // Check for match
                let can_match = src_i + old_len <= len && cnt < max_count;
                let matches = if can_match {
                    let mut m = true;
                    for j in 0..old_len {
                        if unsafe { *src_str.add(src_i + j) } != old_cps[j] {
                            m = false;
                            break;
                        }
                    }
                    m
                } else {
                    false
                };

                if matches {
                    // Copy new string
                    for &cp in &new_cps {
                        unsafe { *dst_str.add(dst_i) = cp; }
                        dst_i += 1;
                    }
                    src_i += old_len;
                    cnt += 1;
                } else {
                    // Copy single character
                    unsafe { *dst_str.add(dst_i) = *src_str.add(src_i); }
                    src_i += 1;
                    dst_i += 1;
                }
            }
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let dst_str = unsafe { dst_base.add(elem_idx * out_max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let mut src_i = 0;
            let mut dst_i = 0;
            let mut cnt = 0usize;

            while src_i < len {
                let can_match = src_i + old_len <= len && cnt < max_count;
                let matches = if can_match {
                    let mut m = true;
                    for j in 0..old_len {
                        if unsafe { *src_str.add(src_i + j) } != old_cps[j] {
                            m = false;
                            break;
                        }
                    }
                    m
                } else {
                    false
                };

                if matches {
                    for &cp in &new_cps {
                        unsafe { *dst_str.add(dst_i) = cp; }
                        dst_i += 1;
                    }
                    src_i += old_len;
                    cnt += 1;
                } else {
                    unsafe { *dst_str.add(dst_i) = *src_str.add(src_i); }
                    src_i += 1;
                    dst_i += 1;
                }
            }
        }
    }

    Some(out)
}

/// Count occurrences of substring in each string.
pub fn count(arr: &RumpyArray, sub: &str) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let sub_cps: Vec<u32> = sub.chars().map(|c| c as u32).collect();
    let sub_len = sub_cps.len();

    let size = arr.size();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), DType::int64());

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst = out_buffer.as_mut_ptr() as *mut i64;

    // Empty substring: return len + 1 for each string (matches numpy)
    if sub_len == 0 {
        if arr.is_c_contiguous() {
            let src_base = arr.data_ptr() as *const u32;
            for elem_idx in 0..size {
                let src_str = unsafe { src_base.add(elem_idx * max_chars) };
                let len = unsafe { utf32_strlen(src_str, max_chars) };
                unsafe { *dst.add(elem_idx) = (len + 1) as i64 };
            }
        } else {
            let src_base = arr.data_ptr();
            for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
                let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
                let len = unsafe { utf32_strlen(src_str, max_chars) };
                unsafe { *dst.add(elem_idx) = (len + 1) as i64 };
            }
        }
        return Some(out);
    }

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            // Count non-overlapping occurrences
            let mut cnt: i64 = 0;
            if len >= sub_len {
                let mut i = 0;
                'outer: while i <= len - sub_len {
                    for j in 0..sub_len {
                        if unsafe { *src_str.add(i + j) } != sub_cps[j] {
                            i += 1;
                            continue 'outer;
                        }
                    }
                    cnt += 1;
                    i += sub_len; // non-overlapping
                }
            }
            unsafe { *dst.add(elem_idx) = cnt };
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let mut cnt: i64 = 0;
            if len >= sub_len {
                let mut i = 0;
                'outer: while i <= len - sub_len {
                    for j in 0..sub_len {
                        if unsafe { *src_str.add(i + j) } != sub_cps[j] {
                            i += 1;
                            continue 'outer;
                        }
                    }
                    cnt += 1;
                    i += sub_len;
                }
            }
            unsafe { *dst.add(elem_idx) = cnt };
        }
    }

    Some(out)
}

/// Get the length of each string (fast direct implementation).
pub fn str_len(arr: &RumpyArray) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let size = arr.size();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), DType::int64());

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst = out_buffer.as_mut_ptr() as *mut i64;

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };
            unsafe { *dst.add(elem_idx) = len as i64 };
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let len = unsafe { utf32_strlen(src_str, max_chars) };
            unsafe { *dst.add(elem_idx) = len as i64 };
        }
    }

    Some(out)
}

// ============================================================================
// Tier 3 - Predicate Operations (Fast UTF-32 implementations)
// ============================================================================

/// Check if each string contains only alphabetic characters.
pub fn isalpha(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_predicate_fast(arr, |cps| !cps.is_empty() && cps.iter().all(|&cp| cp_is_alpha(cp)))
}

/// Check if each string contains only digits.
pub fn isdigit(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_predicate_fast(arr, |cps| !cps.is_empty() && cps.iter().all(|&cp| cp_is_digit(cp)))
}

/// Check if each string contains only alphanumeric characters.
pub fn isalnum(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_predicate_fast(arr, |cps| !cps.is_empty() && cps.iter().all(|&cp| cp_is_alnum(cp)))
}

/// Check if each string is uppercase.
pub fn isupper(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_predicate_fast(arr, |cps| {
        let has_cased = cps.iter().any(|&cp| cp_is_cased(cp));
        has_cased && cps.iter().all(|&cp| !cp_is_lower(cp))
    })
}

/// Check if each string is lowercase.
pub fn islower(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_predicate_fast(arr, |cps| {
        let has_cased = cps.iter().any(|&cp| cp_is_cased(cp));
        has_cased && cps.iter().all(|&cp| !cp_is_upper(cp))
    })
}

/// Check if each string contains only whitespace.
pub fn isspace(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_predicate_fast(arr, |cps| !cps.is_empty() && cps.iter().all(|&cp| cp_is_space(cp)))
}

/// Check if each string starts with the given prefix.
pub fn startswith(arr: &RumpyArray, prefix: &str) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let prefix_cps: Vec<u32> = prefix.chars().map(|c| c as u32).collect();
    let prefix_len = prefix_cps.len();

    let size = arr.size();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), DType::bool());

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst = out_buffer.as_mut_ptr();

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            // Check prefix directly without computing full length
            let mut matches = true;
            for i in 0..prefix_len {
                let cp = unsafe { *src_str.add(i) };
                if cp != prefix_cps[i] {
                    matches = false;
                    break;
                }
            }
            unsafe { *dst.add(elem_idx) = if matches { 1 } else { 0 } };
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let mut matches = true;
            for i in 0..prefix_len {
                let cp = unsafe { *src_str.add(i) };
                if cp != prefix_cps[i] {
                    matches = false;
                    break;
                }
            }
            unsafe { *dst.add(elem_idx) = if matches { 1 } else { 0 } };
        }
    }

    Some(out)
}

/// Check if each string ends with the given suffix.
pub fn endswith(arr: &RumpyArray, suffix: &str) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let suffix_cps: Vec<u32> = suffix.chars().map(|c| c as u32).collect();
    let suffix_len = suffix_cps.len();

    let size = arr.size();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), DType::bool());

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst = out_buffer.as_mut_ptr();

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };
            let matches = if len >= suffix_len {
                let start = len - suffix_len;
                let mut ok = true;
                for i in 0..suffix_len {
                    if unsafe { *src_str.add(start + i) } != suffix_cps[i] {
                        ok = false;
                        break;
                    }
                }
                ok
            } else {
                false
            };
            unsafe { *dst.add(elem_idx) = if matches { 1 } else { 0 } };
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let len = unsafe { utf32_strlen(src_str, max_chars) };
            let matches = if len >= suffix_len {
                let start = len - suffix_len;
                let mut ok = true;
                for i in 0..suffix_len {
                    if unsafe { *src_str.add(start + i) } != suffix_cps[i] {
                        ok = false;
                        break;
                    }
                }
                ok
            } else {
                false
            };
            unsafe { *dst.add(elem_idx) = if matches { 1 } else { 0 } };
        }
    }

    Some(out)
}

/// Check if each string is decimal (contains only decimal characters).
pub fn isdecimal(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_predicate_fast(arr, |cps| !cps.is_empty() && cps.iter().all(|&cp| cp_is_digit(cp)))
}

/// Check if each string is numeric.
pub fn isnumeric(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_predicate_fast(arr, |cps| {
        !cps.is_empty() && cps.iter().all(|&cp| {
            char::from_u32(cp).map_or(false, |c| c.is_numeric())
        })
    })
}

/// Check if each string is titlecase.
pub fn istitle(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_predicate_fast(arr, |cps| {
        if cps.is_empty() {
            return false;
        }
        let mut has_cased = false;
        let mut prev_cased = false;
        for &cp in cps {
            if cp_is_upper(cp) {
                if prev_cased {
                    return false; // Uppercase after cased = not titlecase
                }
                has_cased = true;
                prev_cased = true;
            } else if cp_is_lower(cp) {
                if !prev_cased {
                    return false; // Lowercase at start of word = not titlecase
                }
                has_cased = true;
                prev_cased = true;
            } else {
                prev_cased = false;
            }
        }
        has_cased // Must have at least one cased character
    })
}

// ============================================================================
// Tier 4 - Formatting Operations
// ============================================================================

/// Center each string in a field of given width.
pub fn center(arr: &RumpyArray, width: usize, fillchar: char) -> Option<RumpyArray> {
    apply_unary_string_op(arr, |s| {
        let len = s.chars().count();
        if len >= width {
            s.to_string()
        } else {
            let total_pad = width - len;
            let left_pad = total_pad / 2;
            let right_pad = total_pad - left_pad;
            format!(
                "{}{}{}",
                fillchar.to_string().repeat(left_pad),
                s,
                fillchar.to_string().repeat(right_pad)
            )
        }
    })
}

/// Left-justify each string in a field of given width.
pub fn ljust(arr: &RumpyArray, width: usize, fillchar: char) -> Option<RumpyArray> {
    apply_unary_string_op(arr, |s| {
        let len = s.chars().count();
        if len >= width {
            s.to_string()
        } else {
            format!("{}{}", s, fillchar.to_string().repeat(width - len))
        }
    })
}

/// Right-justify each string in a field of given width.
pub fn rjust(arr: &RumpyArray, width: usize, fillchar: char) -> Option<RumpyArray> {
    apply_unary_string_op(arr, |s| {
        let len = s.chars().count();
        if len >= width {
            s.to_string()
        } else {
            format!("{}{}", fillchar.to_string().repeat(width - len), s)
        }
    })
}

/// Pad each string with zeros on the left to fill width.
pub fn zfill(arr: &RumpyArray, width: usize) -> Option<RumpyArray> {
    apply_unary_string_op(arr, |s| {
        let len = s.chars().count();
        if len >= width {
            s.to_string()
        } else {
            // Handle sign
            let (sign, rest) = if s.starts_with('-') || s.starts_with('+') {
                (&s[0..1], &s[1..])
            } else {
                ("", s)
            };
            format!("{}{}{}", sign, "0".repeat(width - len), rest)
        }
    })
}

/// Capitalize first character of each string (direct UTF-32).
pub fn capitalize(arr: &RumpyArray) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let size = arr.size();
    let dtype = arr.dtype().clone();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), dtype);

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst_base = out_buffer.as_mut_ptr() as *mut u32;

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let dst_str = unsafe { dst_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            if len > 0 {
                // First char: uppercase
                unsafe { *dst_str = codepoint_to_upper(*src_str); }
                // Rest: lowercase
                for i in 1..len {
                    unsafe { *dst_str.add(i) = codepoint_to_lower(*src_str.add(i)); }
                }
            }
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let dst_str = unsafe { dst_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            if len > 0 {
                unsafe { *dst_str = codepoint_to_upper(*src_str); }
                for i in 1..len {
                    unsafe { *dst_str.add(i) = codepoint_to_lower(*src_str.add(i)); }
                }
            }
        }
    }

    Some(out)
}

/// Titlecase each string (direct UTF-32).
pub fn title(arr: &RumpyArray) -> Option<RumpyArray> {
    let max_chars = match arr.dtype().kind() {
        DTypeKind::Str(mc) => mc,
        _ => return None,
    };

    let size = arr.size();
    let dtype = arr.dtype().clone();
    let mut out = RumpyArray::zeros(arr.shape().to_vec(), dtype);

    if size == 0 {
        return Some(out);
    }

    let buffer = out.buffer_mut();
    let out_buffer = Arc::get_mut(buffer).expect("buffer must be unique");
    let dst_base = out_buffer.as_mut_ptr() as *mut u32;

    if arr.is_c_contiguous() {
        let src_base = arr.data_ptr() as *const u32;
        for elem_idx in 0..size {
            let src_str = unsafe { src_base.add(elem_idx * max_chars) };
            let dst_str = unsafe { dst_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let mut prev_cased = false;
            for i in 0..len {
                let cp = unsafe { *src_str.add(i) };
                let is_cased = cp_is_cased(cp);
                let result = if is_cased {
                    if prev_cased {
                        codepoint_to_lower(cp)
                    } else {
                        codepoint_to_upper(cp)
                    }
                } else {
                    cp
                };
                unsafe { *dst_str.add(i) = result; }
                prev_cased = is_cased;
            }
        }
    } else {
        let src_base = arr.data_ptr();
        for (elem_idx, src_offset) in arr.iter_offsets().enumerate() {
            let src_str = unsafe { src_base.offset(src_offset) as *const u32 };
            let dst_str = unsafe { dst_base.add(elem_idx * max_chars) };
            let len = unsafe { utf32_strlen(src_str, max_chars) };

            let mut prev_cased = false;
            for i in 0..len {
                let cp = unsafe { *src_str.add(i) };
                let is_cased = cp_is_cased(cp);
                let result = if is_cased {
                    if prev_cased {
                        codepoint_to_lower(cp)
                    } else {
                        codepoint_to_upper(cp)
                    }
                } else {
                    cp
                };
                unsafe { *dst_str.add(i) = result; }
                prev_cased = is_cased;
            }
        }
    }

    Some(out)
}

/// Swap case of each string (direct UTF-32).
pub fn swapcase(arr: &RumpyArray) -> Option<RumpyArray> {
    apply_codepoint_transform_strided(arr, |cp| {
        if cp_is_upper(cp) {
            codepoint_to_lower(cp)
        } else if cp_is_lower(cp) {
            codepoint_to_upper(cp)
        } else {
            cp
        }
    })
}

// ============================================================================
// Additional Operations
// ============================================================================

/// String comparison mode.
#[derive(Clone, Copy)]
enum StrCmp { Eq, Ne, Lt, Le, Gt, Ge }

/// Compare two string arrays element-wise with given comparison.
fn compare_strings(a: &RumpyArray, b: &RumpyArray, cmp: StrCmp) -> Option<RumpyArray> {
    if a.shape() != b.shape() {
        return None;
    }

    let size = a.size();
    let mut out = RumpyArray::zeros(a.shape().to_vec(), DType::bool());
    let buffer = Arc::get_mut(&mut out.buffer).expect("buffer must be unique");
    let ptr = buffer.as_mut_ptr();

    for i in 0..size {
        let sa = unsafe { read_string(a, i) };
        let sb = unsafe { read_string(b, i) };
        let result = match cmp {
            StrCmp::Eq => sa == sb,
            StrCmp::Ne => sa != sb,
            StrCmp::Lt => sa < sb,
            StrCmp::Le => sa <= sb,
            StrCmp::Gt => sa > sb,
            StrCmp::Ge => sa >= sb,
        };
        unsafe { *ptr.add(i) = if result { 1 } else { 0 } };
    }

    Some(out)
}

/// Compare two string arrays element-wise for equality.
pub fn equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    compare_strings(a, b, StrCmp::Eq)
}

/// Compare two string arrays element-wise for inequality.
pub fn not_equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    compare_strings(a, b, StrCmp::Ne)
}

/// Compare two string arrays element-wise (less than).
pub fn less(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    compare_strings(a, b, StrCmp::Lt)
}

/// Compare two string arrays element-wise (less than or equal).
pub fn less_equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    compare_strings(a, b, StrCmp::Le)
}

/// Compare two string arrays element-wise (greater than).
pub fn greater(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    compare_strings(a, b, StrCmp::Gt)
}

/// Compare two string arrays element-wise (greater than or equal).
pub fn greater_equal(a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    compare_strings(a, b, StrCmp::Ge)
}

/// Expand tabs to spaces.
pub fn expandtabs(arr: &RumpyArray, tabsize: usize) -> Option<RumpyArray> {
    apply_unary_string_op(arr, |s| {
        let mut result = String::new();
        let mut col = 0;
        for c in s.chars() {
            if c == '\t' {
                // Calculate spaces needed to reach next tab stop
                let spaces = if tabsize == 0 { 0 } else { tabsize - (col % tabsize) };
                result.extend(std::iter::repeat(' ').take(spaces));
                col += spaces;
            } else if c == '\n' || c == '\r' {
                result.push(c);
                col = 0;
            } else {
                result.push(c);
                col += 1;
            }
        }
        result
    })
}

/// Join array elements with separator.
pub fn join_with(arr: &RumpyArray, sep: &str) -> Option<String> {
    let size = arr.size();
    let mut strings = Vec::with_capacity(size);
    for i in 0..size {
        strings.push(unsafe { read_string(arr, i) });
    }
    Some(strings.join(sep))
}
