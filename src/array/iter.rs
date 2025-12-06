//! Strided iteration over array elements by byte offset.
//!
//! Uses backstrides for efficient iteration without per-element index calculations.
//! See `designs/backstride-iteration.md` for design rationale.

/// Iterator yielding byte offsets for each element in row-major order.
///
/// Uses backstrides (precomputed wrap distances) for efficient iteration.
/// Only tracks counters for wrap detection, not for offset calculation.
pub struct StridedIter {
    shape: Vec<usize>,
    strides: Vec<isize>,
    backstrides: Vec<isize>,  // (shape[i] - 1) * stride[i]
    counters: Vec<usize>,
    offset: isize,
    remaining: usize,
}

impl StridedIter {
    /// Create a new strided iterator.
    pub fn new(shape: &[usize], strides: &[isize], base_offset: isize) -> Self {
        let size: usize = shape.iter().product();
        let backstrides: Vec<isize> = shape
            .iter()
            .zip(strides.iter())
            .map(|(&s, &st)| if s > 0 { (s as isize - 1) * st } else { 0 })
            .collect();

        Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            backstrides,
            counters: vec![0; shape.len()],
            offset: base_offset,
            remaining: size,
        }
    }

    /// Advance to next element, updating offset in place.
    #[inline]
    fn advance(&mut self) {
        let ndim = self.shape.len();
        for i in (0..ndim).rev() {
            self.counters[i] += 1;
            if self.counters[i] < self.shape[i] {
                self.offset += self.strides[i];
                return;
            }
            // Wrap around using precomputed backstride
            self.offset -= self.backstrides[i];
            self.counters[i] = 0;
        }
    }
}

impl Iterator for StridedIter {
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

impl ExactSizeIterator for StridedIter {}

/// Iterator yielding base byte offsets for axis reduction.
///
/// Iterates over all positions in dimensions OTHER than the reduction axis.
/// Each yielded offset is the starting point for iterating along the axis.
pub struct AxisOffsetIter {
    inner: StridedIter,
}

impl AxisOffsetIter {
    /// Create iterator for axis reduction.
    ///
    /// `shape` and `strides` are from the original array.
    /// `axis` is the reduction axis (will be skipped in iteration).
    pub fn new(shape: &[usize], strides: &[isize], axis: usize, base_offset: isize) -> Self {
        // Build outer shape/strides (excluding axis)
        let mut outer_shape = Vec::with_capacity(shape.len().saturating_sub(1));
        let mut outer_strides = Vec::with_capacity(strides.len().saturating_sub(1));

        for (i, (&s, &st)) in shape.iter().zip(strides.iter()).enumerate() {
            if i != axis {
                outer_shape.push(s);
                outer_strides.push(st);
            }
        }

        // Handle scalar case (all dimensions removed)
        if outer_shape.is_empty() {
            outer_shape.push(1);
            outer_strides.push(0);
        }

        Self {
            inner: StridedIter::new(&outer_shape, &outer_strides, base_offset),
        }
    }
}

impl Iterator for AxisOffsetIter {
    type Item = isize;

    #[inline]
    fn next(&mut self) -> Option<isize> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for AxisOffsetIter {}
