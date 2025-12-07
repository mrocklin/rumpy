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
src/
├── lib.rs                    # Crate root, PyO3 module entry point
├── array/                    # Core array types
│   ├── mod.rs                # RumpyArray struct, constructors, shape ops
│   ├── buffer.rs             # ArrayBuffer (Arc-wrapped memory)
│   ├── flags.rs              # ArrayFlags (contiguity, writeable)
│   ├── iter.rs               # StridedIter, AxisOffsetIter
│   └── dtype/                # DType system
│       ├── mod.rs            # DType wrapper, DTypeOps trait, promotion
│       ├── macros.rs         # impl_float_dtype!, impl_signed_int_dtype!
│       ├── float32.rs        # Float32 implementation
│       ├── float64.rs        # Float64 implementation
│       ├── int*.rs           # Integer type implementations
│       ├── bool.rs           # Boolean implementation
│       └── complex*.rs       # Complex number implementations
│
├── ops/                      # Operations (Rust implementation)
│   ├── mod.rs                # RumpyArray methods, re-exports
│   ├── ufunc.rs              # Core: map_unary_op, map_binary_op, reduce_axis_op
│   ├── registry.rs           # Type-specific optimized loops (SIMD)
│   ├── comparison.rs         # logical_and/or/xor/not, equal, isclose
│   ├── bitwise.rs            # bitwise_and/or/xor/not, shifts
│   ├── statistics.rs         # histogram, cov, corrcoef, median
│   ├── numerical.rs          # gradient, trapezoid, interp, correlate
│   ├── poly.rs               # polyfit, polyval, polyder, polyint, roots
│   ├── set_ops.rs            # isin, intersect1d, union1d, setdiff1d
│   ├── indexing.rs           # take, put, searchsorted, compress
│   ├── linalg.rs             # eig, svd, lstsq, pinv, matrix_rank
│   ├── fft.rs                # FFT operations
│   ├── matmul.rs             # Matrix multiplication
│   ├── dot.rs, inner.rs      # Dot and inner products
│   ├── outer.rs              # Outer product
│   └── solve.rs              # Linear system solving
│
├── python/                   # PyO3 bindings (thin wrappers)
│   ├── mod.rs                # register_module, re-exports
│   ├── pyarray.rs            # PyRumpyArray class, dunder methods
│   ├── creation.rs           # zeros, ones, arange, linspace, eye, full
│   ├── ufuncs.rs             # sqrt, sin, cos, add, multiply (math ops)
│   ├── reductions.rs         # sum, mean, std, nansum, nanmean
│   ├── shape.rs              # reshape, transpose, stack, split, flip
│   ├── indexing.rs           # take, put, searchsorted, compress
│   ├── random.rs             # Generator class, default_rng (submodule)
│   ├── linalg.rs             # linalg submodule bindings
│   └── fft.rs                # fft submodule bindings
│
└── random/                   # Random number generation (Rust)
    ├── mod.rs                # Generator struct
    └── pcg64.rs              # PCG64DXSM implementation

tests/                        # pytest tests (compare against numpy)
├── helpers.py                # assert_eq utility
├── test_creation.py          # Array creation tests
├── test_math.py              # Unary/binary math ops
├── test_reductions.py        # sum, mean, std, etc.
├── test_shape.py             # reshape, transpose, stack
├── test_linalg.py            # Linear algebra
├── test_random.py            # Random number generation
└── ...

designs/                      # Architecture docs (why decisions were made)
plans/                        # Work tracking (what's done, what's next)
```

### Python Bindings Organization

The `src/python/` directory is organized by category to minimize context needed when adding new functions:

| File | Contains | Pattern |
|------|----------|---------|
| `creation.rs` | `zeros`, `ones`, `arange`, `linspace`, `eye`, `full`, `empty`, `*_like` | Thin wrapper calling `RumpyArray::*` |
| `ufuncs.rs` | `sqrt`, `sin`, `add`, `multiply`, `maximum`, `arctan2` | Calls `inner.unary_op()` or `map_binary_op` |
| `reductions.rs` | `sum`, `mean`, `std`, `nansum`, `argmax` | Calls `inner.reduce_*` methods |
| `shape.rs` | `reshape`, `transpose`, `stack`, `split`, `flip` | Shape manipulation wrappers |
| `indexing.rs` | `take`, `put`, `searchsorted`, `where` | Index-based operations |
| `pyarray.rs` | `PyRumpyArray` class, `__add__`, `__getitem__` | Array methods, not module functions |

To add a new function: find the right category file, add the `#[pyfunction]`, then register in `mod.rs`.

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
