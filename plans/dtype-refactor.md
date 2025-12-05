<!-- AI: Read this for dtype refactor work -->

# DType Refactor for Complex & Extensibility

**Design**: See `designs/dtype-system.md`

## Goal

Refactor DTypeOps from f64-based value passing to buffer-based operations.
This enables complex numbers, decimals, and other types that don't fit in f64.

## Architecture Change

**Before**: `read_element() -> f64`, `write_element(f64)` - all values shuttle through f64

**After**:
- Buffer-based ops: `add(a_ptr, b_ptr, out_ptr)` - each dtype works with its native type
- Python interop: `to_pyobject()`, `from_pyobject()` - only universal interface
- No universal Rust value type needed

## Phases

### Phase 1: Refactor DTypeOps trait
- [ ] Add buffer-based binary ops: `add`, `sub`, `mul`, `div`
- [ ] Add buffer-based unary ops: `neg`, `sqrt`, `abs`, etc.
- [ ] Add reduction helpers: `zero_buffer`, `add_acc`
- [ ] Add Python interop: `to_pyobject`, `from_pyobject`
- [ ] Keep `read_element`/`write_element` temporarily for transition

### Phase 2: Update existing dtypes
- [ ] Float64Ops: implement new methods
- [ ] Float32Ops, Int64Ops, Int32Ops, etc.
- [ ] Update operations in `src/ops/mod.rs` to use new methods

### Phase 3: Add complex128
- [ ] Create `src/array/dtype/complex128.rs`
- [ ] Add `DTypeKind::Complex128`
- [ ] Implement all ops working with (f64, f64) pairs
- [ ] Python interop with `complex`
- [ ] Tests comparing against numpy

### Phase 4: Cleanup
- [ ] Remove deprecated `read_element`/`write_element`
- [ ] Update design doc

## Key Files

- `src/array/dtype/mod.rs` - DTypeOps trait
- `src/array/dtype/*.rs` - dtype implementations
- `src/ops/mod.rs` - map_unary, map_binary, reduce_*
- `src/array/mod.rs` - constructors, get_element
