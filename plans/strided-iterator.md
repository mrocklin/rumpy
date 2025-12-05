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

### Phase 1: Create StridedIter
- [ ] Add `src/array/iter.rs` with `StridedIter` struct
- [ ] Implement `Iterator` trait
- [ ] Add `iter_offsets()` method to `RumpyArray`
- [ ] Add contiguous fast path (just increment by itemsize)

### Phase 2: Convert simple cases in ops/mod.rs
Start with straightforward loops:
- [ ] `map_compare` (~line 396)
- [ ] `all` / `any` (~line 1419)
- [ ] `count_nonzero` (~line 1343)
- [ ] `clip` (~line 1549)

### Phase 3: Convert remaining ops/mod.rs
- [ ] `variance` / `std` calculations
- [ ] `argmax` / `argmin`
- [ ] `sort` / `argsort`
- [ ] Reduction axis loops

### Phase 4: Convert other files
- [ ] `src/array/mod.rs` - select_by_mask, select_by_indices, astype
- [ ] `src/ops/fft.rs`
- [ ] `src/ops/linalg.rs`
- [ ] `src/python/mod.rs`

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
