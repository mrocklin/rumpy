//! Strided iteration over array elements by byte offset.

/// Iterator yielding byte offsets for each element in row-major order.
///
/// More efficient than the `increment_indices` pattern because it
/// advances by strides directly rather than computing byte offset
/// from indices on each iteration.
pub struct StridedIter<'a> {
    shape: &'a [usize],
    strides: &'a [isize],
    indices: Vec<usize>,
    offset: isize,
    remaining: usize,
}

impl<'a> StridedIter<'a> {
    /// Create a new strided iterator.
    pub fn new(shape: &'a [usize], strides: &'a [isize], base_offset: isize) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            strides,
            indices: vec![0; shape.len()],
            offset: base_offset,
            remaining: size,
        }
    }

    /// Advance to next element, updating offset in place.
    #[inline]
    fn advance(&mut self) {
        let ndim = self.shape.len();
        for i in (0..ndim).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.shape[i] {
                self.offset += self.strides[i];
                return;
            }
            // Wrap around: subtract the distance we traveled in this dimension
            self.offset -= (self.shape[i] as isize - 1) * self.strides[i];
            self.indices[i] = 0;
        }
    }
}

impl Iterator for StridedIter<'_> {
    type Item = isize;

    #[inline]
    fn next(&mut self) -> Option<isize> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let result = self.offset;
        if self.remaining > 0 {
            self.advance();
        }
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StridedIter<'_> {}
