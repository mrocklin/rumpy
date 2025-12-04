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
    r = rumpy.some_op(...)
    n = np.some_op(...)
    assert_eq(r, n)
```

## File Structure

```
src/array/          # Core Rust types
  mod.rs            # RumpyArray struct, constructors, views, broadcast
  dtype.rs          # DType enum (f32, f64, i32, i64, bool)
  buffer.rs         # ArrayBuffer (Arc-wrapped memory)
  flags.rs          # ArrayFlags (contiguity, writeable)

src/ops/            # Operations (ufunc-style)
  mod.rs            # map_unary, map_binary, reduce_all, reduce_axis

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
- **DType as enum** (not generic) - simpler PyO3 interop
- **Signed strides (isize)** - enables negative strides for reversed views
- **`__array_interface__`** for NumPy interop - zero-copy when possible

## Adding New Operations

Use the ufunc machinery in `src/ops/mod.rs`:

```rust
// Element-wise unary: use map_unary
pub fn sqrt(&self) -> RumpyArray {
    map_unary(self, |x| x.sqrt())
}

// Element-wise binary: use map_binary (handles broadcasting)
pub fn binary_op(&self, other: &RumpyArray, op: BinaryOp) -> Option<RumpyArray> {
    map_binary(self, other, |a, b| a + b)
}

// Full reduction: use reduce_all
pub fn sum(&self) -> f64 {
    reduce_all(self, 0.0, |acc, x| acc + x)
}

// Axis reduction: use reduce_axis
pub fn sum_axis(&self, axis: usize) -> RumpyArray {
    reduce_axis(self, axis, 0.0, |acc, x| acc + x)
}
```

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
