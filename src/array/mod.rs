pub mod buffer;
pub mod dtype;
pub mod flags;

use std::sync::Arc;

pub use buffer::ArrayBuffer;
pub use dtype::DType;
pub use flags::ArrayFlags;

/// Core N-dimensional array type.
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

    /// Check if array is C-contiguous.
    pub fn is_c_contiguous(&self) -> bool {
        self.flags.contains(ArrayFlags::C_CONTIGUOUS)
    }

    /// Check if array is F-contiguous.
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.contains(ArrayFlags::F_CONTIGUOUS)
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
