# Plan: Backstride Iteration

**Design**: `designs/backstride-iteration.md`

## Goal

Replace slow index-based iteration with backstride-based iteration. Target: axis reductions from 5-10x slower than NumPy to ~1-2x.

## Phase 1: BackstrideIter

Replace `StridedIter` in `src/array/iter.rs`:

1. Add `backstrides` field (precomputed on construction)
2. Remove `indices` field dependency for offset calculation
3. Keep counters only for wrap detection
4. Same Iterator interface, drop-in replacement

**Test**: Existing tests should pass unchanged.

## Phase 2: axis_offsets()

Add method to RumpyArray for axis iteration:

```rust
impl RumpyArray {
    /// Iterate over base offsets for reduction along `axis`.
    /// Each offset is the start of a slice along the axis.
    pub fn axis_offsets(&self, axis: usize) -> AxisOffsetIter
}
```

Implementation:
- Build shape/strides for "outer" dimensions (all except axis)
- Use BackstrideIter internally
- Yield byte offsets

## Phase 3: Optimize reduce_axis_op

In `src/ops/mod.rs`, change `reduce_axis_op()`:

Before:
```rust
for j in 0..axis_len {
    in_indices[axis] = j;
    let src_offset = arr.byte_offset_for(&in_indices);  // SLOW
    acc_fn(result_ptr, i, src_ptr, src_offset);
}
```

After:
```rust
let axis_stride = arr.strides()[axis];
for base_offset in arr.axis_offsets(axis) {
    let mut ptr = src_ptr.byte_offset(base_offset);
    for _ in 0..axis_len {
        acc_fn(result_ptr, i, ptr, 0);  // offset=0, ptr already positioned
        ptr = ptr.byte_offset(axis_stride);
    }
}
```

**Contiguous fast path**: When `axis_stride == itemsize`, use slice iteration for SIMD.

## Phase 4: Migrate Other Axis Operations

Apply same pattern to:
- `moment_axis()` (lines 817-859)
- `all_axis()` / `any_axis()` (lines 1563-1667)
- `sort()` / `argsort()` with axis (lines 1345-1472)
- `cumulative_op()` with axis (lines 1728-1800)
- `arg_reduce_axis()` (lines 966-1018)

## Phase 5: Benchmarks

Add benchmarks comparing:
- axis reduction (sum_axis, var_axis) before/after
- Contiguous vs strided arrays
- Compare against NumPy

## Status

- [x] Phase 1: BackstrideIter
- [x] Phase 2: axis_offsets()
- [x] Phase 3: reduce_axis_op optimization
- [x] Phase 3b: Strided reduce loops in registry
- [x] Phase 3c: Multi-accumulator optimization for sum/max/min
- [ ] Phase 4: Migrate other axis operations (deferred)
- [ ] Phase 5: Formal benchmarks (deferred)

## Results

**Before**: axis reductions were 5-10x slower than NumPy (debug build: 116x slower)

**After all optimizations** (release build, 1000x1000 f64 array):

| Operation | Contiguous Axis | Non-contiguous Axis |
|-----------|-----------------|---------------------|
| sum | **0.84x** (faster!) | 3.5x slower |
| prod | **0.96x** (parity) | - |
| max | 1.3x slower | 3.6x slower |
| min | 1.3x slower | - |

**Integer types** (contiguous axis):
| dtype | rumpy/numpy |
|-------|-------------|
| int64 | 0.97x (parity) |
| int32 | 0.30x (3x faster!) |

## Key Techniques

1. **Strided reduce loops**: Process N elements per call, not 1
2. **Multi-accumulator**: 8 independent accumulators break dependency chain for SIMD
3. **Contiguous detection**: `stride == itemsize` triggers slice-based fast path

## Remaining Gap

Non-contiguous axis reductions (3.5x slower) are limited by:
- Cache misses (stride = 8000 bytes for 1000-column array)
- No SIMD across non-contiguous memory

Potential fixes: cache-aware blocking, transpose when profitable.
