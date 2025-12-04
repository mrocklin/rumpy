/// Shared, reference-counted raw data buffer.
pub struct ArrayBuffer {
    data: Box<[u8]>,
}

impl ArrayBuffer {
    /// Create a new zeroed buffer of given size in bytes.
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size].into_boxed_slice(),
        }
    }

    /// Get raw pointer to buffer data.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Get mutable raw pointer to buffer data.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    /// Get buffer length in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
