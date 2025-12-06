# Strided Iterator Abstraction

**Design doc:** `designs/strided-loops.md`

## Problem

Many operations use the slow `increment_indices` pattern:
```rust
let mut indices = vec![0usize; arr.ndim()];  // heap alloc per call
for i in 0..size {
    let offset = arr.byte_offset_for(&indices);  // O(ndim) per element
    // ... do work ...
    increment_indices(&mut indices, arr.shape());  // O(ndim) per element
}
```

The registry already has strided loops for ufuncs (`map_unary_op`, `map_binary_op`), but ~50 other call sites still use the slow pattern.

## Solution

Create a reusable `StridedIter` that:
1. Iterates by advancing byte offsets directly
2. Avoids per-element index computation
3. Provides a clean API for all operations

## Proposed API

```rust
// In src/array/iter.rs

/// Iterator yielding byte offsets for each element
pub struct StridedIter<'a> {
    shape: &'a [usize],
    strides: &'a [isize],
    indices: Vec<usize>,
    offset: isize,
    remaining: usize,
}

impl<'a> StridedIter<'a> {
    pub fn new(arr: &'a RumpyArray) -> Self { ... }
}

impl Iterator for StridedIter<'_> {
    type Item = isize;  // byte offset
    fn next(&mut self) -> Option<isize> { ... }
}

// Convenience on RumpyArray
impl RumpyArray {
    pub fn iter_offsets(&self) -> StridedIter<'_> { ... }
}
```

## Usage Pattern

Before:
```rust
let mut indices = vec![0usize; arr.ndim()];
for i in 0..size {
    let offset = arr.byte_offset_for(&indices);
    unsafe { /* use offset */ }
    increment_indices(&mut indices, arr.shape());
}
```

After:
```rust
for offset in arr.iter_offsets() {
    unsafe { /* use offset */ }
}
```

## Implementation Phases

### Phase 1: Create StridedIter ✓
- [x] Add `src/array/iter.rs` with `StridedIter` struct
- [x] Implement `Iterator` trait with inner-loop optimization
- [x] Add `iter_offsets()` method to `RumpyArray`

### Phase 2: Convert single-array operations ✓
- [x] `all` / `any` / `count_nonzero`
- [x] `clip` / `round`
- [x] `var` / `argmax` / `argmin`
- [x] `to_vec` / `real` / `imag` / `conj`
- [x] `reduce_all_op` (both registry and fallback paths)
- [x] `map_unary_op` fallback
- [x] `cumulative_op` flattened case
- [x] `astype`

### Remaining (~20 usages)
Axis-specific or multi-array operations that need output indices:
- Axis reductions, cumulative ops along axis
- Binary ops with broadcasting (would need `StridedIterZip`)
- `select_by_mask`, `select_by_indices`

## Testing

Run full test suite after each phase:
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
pytest tests/ -v
```

## Notes

- The registry strided loops in `ops/registry.rs` are a different pattern (function pointers with stride args) - those are fine as-is
- Consider also adding `StridedIterMut` for in-place operations
- For multi-array iteration (binary ops), may need `StridedIterZip`
