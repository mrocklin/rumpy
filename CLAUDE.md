# Rumpy Development Guide

NumPy reimplementation in Rust with PyO3 bindings.

## Build & Test

```bash
# First time setup
uv venv && source .venv/bin/activate
uv pip install pytest numpy hypothesis

# Build (required after Rust changes)
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop

# Test
pytest tests/ -v
```

## Testing Philosophy

- **TDD**: Write tests first, compare against numpy
- **Pattern**: Do same operation in rumpy and numpy, use `assert_eq(r, n)`
- **Location**: `tests/helpers.py` has `assert_eq` utility

```python
def test_feature():
    r = rp.some_op(...)
    n = np.some_op(...)
    assert_eq(r, n)
```

## File Structure

```
src/array/          # Core Rust types
  mod.rs            # RumpyArray struct, constructors, views, broadcast
  dtype/            # DType system with macro-generated implementations
    mod.rs          # DType wrapper, DTypeOps trait, type promotion
    macros.rs       # impl_float_dtype!, impl_signed_int_dtype!, etc.
  buffer.rs         # ArrayBuffer (Arc-wrapped memory)
  flags.rs          # ArrayFlags (contiguity, writeable)

src/ops/            # Operations (ufunc-style)
  mod.rs            # RumpyArray methods, error types, re-exports
  ufunc.rs          # Core: map_unary_op, map_binary_op, reduce_axis_op
  registry.rs       # Type-specific optimized loops (SIMD fast paths)
  statistics.rs     # histogram, cov, corrcoef, median, ptp, average
  comparison.rs     # logical_and/or/xor/not, equal, isclose, etc.
  bitwise.rs        # bitwise_and/or/xor/not, left_shift, right_shift

src/python/         # PyO3 bindings
  mod.rs            # Module-level functions (zeros, ones, arange)
  pyarray.rs        # PyRumpyArray class, __add__, reductions

tests/              # pytest tests
  helpers.py        # assert_eq utility
  test_*.py         # Test files

designs/            # Architecture docs (why)
plans/              # Work status (what's next)
```

## Before Starting Work

1. Read `plans/current.md` for current status
2. Read relevant `designs/*.md` for architecture context
3. Run `pytest tests/ -v` to verify working state
4. Check what phase we're in and what's next

## Key Design Decisions

- **Arc<ArrayBuffer>** for views - shared ownership, no copy
- **DType as trait object** - `DType(Arc<dyn DTypeOps>)` with macro-generated impls
- **Two-tier dispatch** - registry fast path → trait fallback
- **Signed strides (isize)** - enables negative strides for reversed views
- **`__array_interface__`** for NumPy interop - zero-copy when possible

## Key Design Docs

| Doc | When to Read |
|-----|--------------|
| `designs/dtype-system.md` | Adding new dtypes |
| `designs/adding-operations.md` | Adding unary/binary/reduce ops |
| `designs/ufuncs.md` | Understanding ufunc architecture |
| `designs/iteration-performance.md` | Optimizing loops, benchmarks |
| `designs/gufuncs.md` | Matrix ops like matmul |
| `designs/linalg.md` | faer integration for linear algebra |
| `designs/deviations.md` | Where rumpy differs from NumPy |

## Adding New Operations

Use the ufunc machinery in `src/ops/ufunc.rs`:

```rust
// Element-wise unary: use map_unary_op with UnaryOp enum
pub fn sqrt(&self) -> Result<RumpyArray, UnaryOpError> {
    map_unary_op(self, UnaryOp::Sqrt)
}

// Element-wise binary: use map_binary_op with BinaryOp enum (handles broadcasting)
pub fn add(&self, other: &RumpyArray) -> Result<RumpyArray, BinaryOpError> {
    map_binary_op(self, other, BinaryOp::Add)
}

// Full reduction: use reduce_all_op with ReduceOp enum
pub fn sum(&self) -> f64 {
    reduce_all_op(self, ReduceOp::Sum, false)  // false = not NaN-aware
}

// Axis reduction: use reduce_axis_op
pub fn sum_axis(&self, axis: usize) -> RumpyArray {
    reduce_axis_op(self, axis, ReduceOp::Sum, false)
}
```

The registry in `src/ops/registry.rs` provides optimized type-specific loops.
To add a new operation, add the enum variant and register loops for each dtype.

Then add Python bindings in `pyarray.rs` and tests.

## Common Pitfalls

- **Empty arrays**: Always check `size == 0` before buffer operations
- **Build command**: Must use `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` for Python 3.14+
- **Closures in match**: Each arm has different type; call function inside each arm instead of returning closure
- **Buffer access**: Use `Arc::get_mut()` only on freshly created arrays

## Adding New Features

1. Add Rust implementation in `src/ops/` (operations) or `src/array/` (core)
2. Add Python bindings in `src/python/pyarray.rs`
3. Export from `src/python/mod.rs` if module-level function
4. Add tests comparing against numpy
5. Update `plans/current.md`
