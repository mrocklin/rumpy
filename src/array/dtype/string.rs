//! String dtype implementations.
//!
//! NumPy has two string types:
//! - Unicode strings (`<Un` / dtype kind 'U'): Fixed-length UTF-32, n chars, 4n bytes
//! - Byte strings (`|Sn` / dtype kind 'S'): Fixed-length bytes, n bytes
//!
//! We implement Unicode strings as UTF-32 (matching NumPy's representation).

use super::{BinaryOp, BitwiseOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};
use std::cmp::Ordering;

/// Unicode string dtype operations.
/// Each string is stored as fixed-length UTF-32 (4 bytes per char).
pub struct StrOps {
    /// Maximum number of characters in each string element.
    pub max_chars: usize,
}

impl StrOps {
    /// Create a new string dtype with the given max character count.
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }

    /// Read a string from buffer at the given byte offset.
    /// Returns the string without trailing null characters.
    ///
    /// # Safety
    /// Pointer must be valid for reading max_chars u32 values at byte_offset.
    #[inline]
    unsafe fn read_string(&self, ptr: *const u8, byte_offset: isize) -> String {
        let start = ptr.offset(byte_offset) as *const u32;
        let mut chars = Vec::with_capacity(self.max_chars);
        for i in 0..self.max_chars {
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

    /// Write a string to buffer at element index.
    /// Truncates if string is too long, pads with nulls if too short.
    ///
    /// # Safety
    /// Pointer must be valid for writing max_chars u32 values at idx.
    #[inline]
    unsafe fn write_string(&self, ptr: *mut u8, idx: usize, s: &str) {
        let start = (ptr as *mut u32).add(idx * self.max_chars);
        let mut written = 0;
        for c in s.chars().take(self.max_chars) {
            *start.add(written) = c as u32;
            written += 1;
        }
        // Pad with nulls
        for i in written..self.max_chars {
            *start.add(i) = 0;
        }
    }

    /// Write a string at byte offset.
    ///
    /// # Safety
    /// Pointer must be valid.
    #[inline]
    unsafe fn write_string_at_offset(&self, ptr: *mut u8, byte_offset: isize, s: &str) {
        let start = ptr.offset(byte_offset) as *mut u32;
        let mut written = 0;
        for c in s.chars().take(self.max_chars) {
            *start.add(written) = c as u32;
            written += 1;
        }
        for i in written..self.max_chars {
            *start.add(i) = 0;
        }
    }
}

impl DTypeOps for StrOps {
    fn kind(&self) -> DTypeKind {
        DTypeKind::Str(self.max_chars)
    }

    fn itemsize(&self) -> usize {
        self.max_chars * 4 // 4 bytes per char (UTF-32)
    }

    fn typestr(&self) -> &'static str {
        // NumPy uses <Un where n is max chars
        // We return a generic placeholder; actual formatting happens in Python bindings
        "<U0"
    }

    fn format_char(&self) -> &'static str {
        // Buffer protocol format for unicode string
        "w" // wchar_t array
    }

    fn name(&self) -> &'static str {
        "str"
    }

    fn promotion_priority(&self) -> u8 {
        0 // Strings don't participate in numeric promotion
    }

    fn is_integer(&self) -> bool {
        false
    }

    unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
        // Zero is an empty string (all null chars)
        let start = (ptr as *mut u32).add(idx * self.max_chars);
        for i in 0..self.max_chars {
            *start.add(i) = 0;
        }
    }

    unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
        // "1" string
        self.write_string(ptr, idx, "1");
    }

    unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
        let s = self.read_string(src, byte_offset);
        self.write_string(dst, idx, &s);
    }

    unsafe fn unary_op(
        &self,
        _op: UnaryOp,
        _src: *const u8,
        _byte_offset: isize,
        _out: *mut u8,
        _idx: usize,
    ) {
        // String types don't support numeric unary ops
        // The char module provides string-specific operations
    }

    unsafe fn binary_op(
        &self,
        _op: BinaryOp,
        _a: *const u8,
        _a_offset: isize,
        _b: *const u8,
        _b_offset: isize,
        _out: *mut u8,
        _idx: usize,
    ) {
        // String types don't support numeric binary ops
        // char.add handles string concatenation
    }

    unsafe fn reduce_init(&self, _op: ReduceOp, ptr: *mut u8, idx: usize) {
        // Initialize with empty string
        self.write_zero(ptr, idx);
    }

    unsafe fn reduce_acc(
        &self,
        _op: ReduceOp,
        _acc: *mut u8,
        _idx: usize,
        _val: *const u8,
        _byte_offset: isize,
    ) {
        // String types don't support numeric reductions
    }

    unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
        let s = self.read_string(ptr, byte_offset);
        format!("'{}'", s)
    }

    unsafe fn compare_elements(
        &self,
        a: *const u8,
        a_offset: isize,
        b: *const u8,
        b_offset: isize,
    ) -> Ordering {
        let sa = self.read_string(a, a_offset);
        let sb = self.read_string(b, b_offset);
        sa.cmp(&sb)
    }

    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
        // Non-empty string is truthy
        let first_char = *(ptr.offset(byte_offset) as *const u32);
        first_char != 0
    }

    unsafe fn write_f64(&self, _ptr: *mut u8, _idx: usize, _val: f64) -> bool {
        false // Strings don't support f64 conversion
    }

    unsafe fn read_f64(&self, _ptr: *const u8, _byte_offset: isize) -> Option<f64> {
        None
    }

    unsafe fn read_i64(&self, _ptr: *const u8, _byte_offset: isize) -> Option<i64> {
        None
    }

    unsafe fn write_i64(&self, _ptr: *mut u8, _idx: usize, _val: i64) -> bool {
        false
    }

    unsafe fn write_f64_at_byte_offset(
        &self,
        _ptr: *mut u8,
        _byte_offset: isize,
        _val: f64,
    ) -> bool {
        false
    }

    unsafe fn write_complex(&self, _ptr: *mut u8, _idx: usize, _real: f64, _imag: f64) -> bool {
        false
    }

    unsafe fn read_complex(&self, _ptr: *const u8, _byte_offset: isize) -> Option<(f64, f64)> {
        None
    }

    unsafe fn bitwise_op(
        &self,
        _op: BitwiseOp,
        _a: *const u8,
        _a_offset: isize,
        _b: *const u8,
        _b_offset: isize,
        _out: *mut u8,
        _idx: usize,
    ) -> bool {
        false
    }

    unsafe fn bitwise_not(
        &self,
        _src: *const u8,
        _byte_offset: isize,
        _out: *mut u8,
        _idx: usize,
    ) -> bool {
        false
    }
}

/// Byte string dtype operations.
/// Each string is stored as fixed-length bytes.
pub struct BytesOps {
    /// Maximum number of bytes in each string element.
    pub max_bytes: usize,
}

impl BytesOps {
    /// Create a new bytes dtype with the given max byte count.
    pub fn new(max_bytes: usize) -> Self {
        Self { max_bytes }
    }

    /// Read bytes from buffer at the given byte offset.
    /// Returns bytes without trailing null bytes.
    ///
    /// # Safety
    /// Pointer must be valid for reading max_bytes bytes at byte_offset.
    #[inline]
    unsafe fn read_bytes(&self, ptr: *const u8, byte_offset: isize) -> Vec<u8> {
        let start = ptr.offset(byte_offset);
        let mut bytes = Vec::with_capacity(self.max_bytes);
        for i in 0..self.max_bytes {
            let b = *start.add(i);
            if b == 0 {
                break;
            }
            bytes.push(b);
        }
        bytes
    }

    /// Write bytes to buffer at element index.
    /// Truncates if too long, pads with null bytes if too short.
    ///
    /// # Safety
    /// Pointer must be valid for writing max_bytes bytes at idx.
    #[inline]
    unsafe fn write_bytes(&self, ptr: *mut u8, idx: usize, data: &[u8]) {
        let start = ptr.add(idx * self.max_bytes);
        let len = data.len().min(self.max_bytes);
        std::ptr::copy_nonoverlapping(data.as_ptr(), start, len);
        // Pad with nulls
        for i in len..self.max_bytes {
            *start.add(i) = 0;
        }
    }

    /// Write bytes at byte offset.
    ///
    /// # Safety
    /// Pointer must be valid.
    #[inline]
    unsafe fn write_bytes_at_offset(&self, ptr: *mut u8, byte_offset: isize, data: &[u8]) {
        let start = ptr.offset(byte_offset);
        let len = data.len().min(self.max_bytes);
        std::ptr::copy_nonoverlapping(data.as_ptr(), start, len);
        for i in len..self.max_bytes {
            *start.add(i) = 0;
        }
    }
}

impl DTypeOps for BytesOps {
    fn kind(&self) -> DTypeKind {
        DTypeKind::Bytes(self.max_bytes)
    }

    fn itemsize(&self) -> usize {
        self.max_bytes
    }

    fn typestr(&self) -> &'static str {
        "|S0"
    }

    fn format_char(&self) -> &'static str {
        "s" // char array
    }

    fn name(&self) -> &'static str {
        "bytes"
    }

    fn promotion_priority(&self) -> u8 {
        0
    }

    fn is_integer(&self) -> bool {
        false
    }

    unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
        let start = ptr.add(idx * self.max_bytes);
        for i in 0..self.max_bytes {
            *start.add(i) = 0;
        }
    }

    unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
        self.write_bytes(ptr, idx, b"1");
    }

    unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
        let data = self.read_bytes(src, byte_offset);
        self.write_bytes(dst, idx, &data);
    }

    unsafe fn unary_op(
        &self,
        _op: UnaryOp,
        _src: *const u8,
        _byte_offset: isize,
        _out: *mut u8,
        _idx: usize,
    ) {
    }

    unsafe fn binary_op(
        &self,
        _op: BinaryOp,
        _a: *const u8,
        _a_offset: isize,
        _b: *const u8,
        _b_offset: isize,
        _out: *mut u8,
        _idx: usize,
    ) {
    }

    unsafe fn reduce_init(&self, _op: ReduceOp, ptr: *mut u8, idx: usize) {
        self.write_zero(ptr, idx);
    }

    unsafe fn reduce_acc(
        &self,
        _op: ReduceOp,
        _acc: *mut u8,
        _idx: usize,
        _val: *const u8,
        _byte_offset: isize,
    ) {
    }

    unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
        let data = self.read_bytes(ptr, byte_offset);
        format!("b'{}'", String::from_utf8_lossy(&data))
    }

    unsafe fn compare_elements(
        &self,
        a: *const u8,
        a_offset: isize,
        b: *const u8,
        b_offset: isize,
    ) -> Ordering {
        let da = self.read_bytes(a, a_offset);
        let db = self.read_bytes(b, b_offset);
        da.cmp(&db)
    }

    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
        let first_byte = *ptr.offset(byte_offset);
        first_byte != 0
    }

    unsafe fn write_f64(&self, _ptr: *mut u8, _idx: usize, _val: f64) -> bool {
        false
    }

    unsafe fn read_f64(&self, _ptr: *const u8, _byte_offset: isize) -> Option<f64> {
        None
    }

    unsafe fn read_i64(&self, _ptr: *const u8, _byte_offset: isize) -> Option<i64> {
        None
    }

    unsafe fn write_i64(&self, _ptr: *mut u8, _idx: usize, _val: i64) -> bool {
        false
    }

    unsafe fn write_f64_at_byte_offset(
        &self,
        _ptr: *mut u8,
        _byte_offset: isize,
        _val: f64,
    ) -> bool {
        false
    }

    unsafe fn write_complex(&self, _ptr: *mut u8, _idx: usize, _real: f64, _imag: f64) -> bool {
        false
    }

    unsafe fn read_complex(&self, _ptr: *const u8, _byte_offset: isize) -> Option<(f64, f64)> {
        None
    }

    unsafe fn bitwise_op(
        &self,
        _op: BitwiseOp,
        _a: *const u8,
        _a_offset: isize,
        _b: *const u8,
        _b_offset: isize,
        _out: *mut u8,
        _idx: usize,
    ) -> bool {
        false
    }

    unsafe fn bitwise_not(
        &self,
        _src: *const u8,
        _byte_offset: isize,
        _out: *mut u8,
        _idx: usize,
    ) -> bool {
        false
    }
}

// Helper functions for string operations (used by char module)
impl StrOps {
    /// Read string at byte offset (public for char module).
    ///
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn read_str(&self, ptr: *const u8, byte_offset: isize) -> String {
        self.read_string(ptr, byte_offset)
    }

    /// Write string at element index (public for char module).
    ///
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn write_str(&self, ptr: *mut u8, idx: usize, s: &str) {
        self.write_string(ptr, idx, s);
    }

    /// Write string at byte offset (public for char module).
    ///
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn write_str_at_offset(&self, ptr: *mut u8, byte_offset: isize, s: &str) {
        self.write_string_at_offset(ptr, byte_offset, s);
    }
}

impl BytesOps {
    /// Read bytes at byte offset (public for char module).
    ///
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn read_data(&self, ptr: *const u8, byte_offset: isize) -> Vec<u8> {
        self.read_bytes(ptr, byte_offset)
    }

    /// Write bytes at element index (public for char module).
    ///
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn write_data(&self, ptr: *mut u8, idx: usize, data: &[u8]) {
        self.write_bytes(ptr, idx, data);
    }

    /// Write bytes at byte offset (public for char module).
    ///
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn write_data_at_offset(&self, ptr: *mut u8, byte_offset: isize, data: &[u8]) {
        self.write_bytes_at_offset(ptr, byte_offset, data);
    }
}
