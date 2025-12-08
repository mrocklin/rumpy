# f64 Generalization Plan

**Design doc:** `designs/kernel-dispatch.md`
**Goal:** Replace f64-hardcoded operations with kernel/dispatch for dtype-preserving performance.

## Problem

Several operations use `read_f64`/`write_f64` trait fallback instead of kernel/dispatch:
- Converts all dtypes to f64, operates, converts back
- Performance penalty for non-f64 types
- Loses precision for int64 > 2^53

## High Priority

### Phase 1: clip
Location: `src/ops/mod.rs:221-251`

Current:
```rust
let val = ops.read_f64(src_ptr, offset).unwrap_or(0.0);
// clamp logic
ops.write_f64(result_ptr, i, val);
```

Plan:
- [ ] Add `Clip` struct to `kernels/arithmetic.rs` (parameterized by min/max)
- [ ] Add `dispatch_clip` in `dispatch.rs`
- [ ] Replace `RumpyArray::clip()` to use dispatch
- [ ] Test with int32, int64, float32, float64

### Phase 2: round
Location: `src/ops/mod.rs:254-276`

Current: Same read_f64/write_f64 pattern with scale multiply.

Plan:
- [ ] Add `Round` kernel (parameterized by decimals/scale)
- [ ] Add `dispatch_round` in `dispatch.rs`
- [ ] Replace `RumpyArray::round()`
- [ ] Test precision preservation

### Phase 3: where_
Location: `src/ops/mod.rs:356-383`

Current: Uses read_f64/write_f64 for value selection.

Plan:
- [ ] Add `dispatch_where` in `dispatch.rs`
- [ ] Handle mixed dtypes via promotion
- [ ] Test with various dtype combinations

## Medium Priority

### Phase 4: cumsum/cumprod axis
Location: `src/ops/array_methods/cumulative.rs:194-196`

Uses `get_element` in loop - O(ndim) per element.

Plan:
- [ ] Add `CumSum`, `CumProd` to reduce kernels
- [ ] Add axis dispatch similar to reduce_axis

### Phase 5: all/any axis
Location: `src/ops/array_methods/logical.rs:77, 140`

Uses `get_element` in loop.

Plan:
- [ ] Add `All`, `Any` reduce kernels (bool output)
- [ ] Add dispatch path

### Phase 6: NaN reductions
Location: `src/ops/array_methods/reductions.rs:627-893`

Many use read_f64 fallback.

Plan:
- [ ] Add `NanSum`, `NanProd`, `NanMax`, `NanMin` kernels
- [ ] Wire through dispatch

## Not Planned (Acceptable)

- `get_element()/set_element()` returning f64 - API convenience
- `sum()/mean()` returning f64 - matches NumPy scalar behavior
- Linear algebra (faer requires f64)
- FFT (inherently complex128)
- Random (output dtype is separate from generation)

## Testing Strategy

Each phase:
1. Verify existing tests pass
2. Add dtype-specific tests (int32, int64, float32)
3. Benchmark before/after for large arrays

## Current Status

**All phases complete.** Added typed dispatch for:

- [x] Phase 1: clip - Added `dispatch_clip` in dispatch.rs
- [x] Phase 2: round - Added `dispatch_round` with f64/f32 and int variants
- [x] Phase 3: where_ - Added `dispatch_where` with typed path
- [x] Phase 4: cumsum/cumprod - Added `dispatch_cumsum`, `dispatch_cumprod` with typed path + `loops::cumulative`
- [x] Phase 5: all/any axis - Added `dispatch_all_axis`, `dispatch_any_axis` with typed path
- [x] Phase 6: NaN reductions - Already have typed dispatch (dispatch_nan_reduce_axis_*)

Remaining usages of `get_element`/`read_f64` are:
- Fallback paths for unsupported dtypes (Bool, DateTime, Float16)
- Non-performance-critical operations (sorting, to_vec)
- Full-array logical ops (all(), any(), count_nonzero()) - acceptable
