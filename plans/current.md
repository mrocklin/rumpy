<!-- AI: Read this before starting work on rumpy -->

# Current Development Status

**Last updated**: Phase 3 complete

## What's Done

- [x] Project scaffolding (Cargo.toml, pyproject.toml, maturin)
- [x] Core types: DType, ArrayFlags, ArrayBuffer, RumpyArray
- [x] Basic constructors: zeros(), ones(), arange()
- [x] Python bindings with __array_interface__
- [x] Test infrastructure with assert_eq helper
- [x] Views: view_with(), slicing, reshape, transpose
- [x] Integer indexing: arr[i], arr[i, j] returns scalar
- [x] All 55 tests passing

## Key Files

- `src/array/mod.rs` - RumpyArray struct, constructors, views, arange
- `src/array/dtype.rs` - DType enum
- `src/python/pyarray.rs` - Python class with __getitem__, reshape, T
- `src/python/mod.rs` - Module-level functions (zeros, ones, arange)
- `tests/helpers.py` - assert_eq for comparing against numpy

## Building

```bash
# Create venv and install
uv venv && source .venv/bin/activate
uv pip install pytest numpy hypothesis
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop

# Run tests
pytest tests/ -v
```

## Next Phase: Binary Operations

Priority order:
1. Element-wise add, sub, mul, div (same shape)
2. Scalar operations (arr + 5)
3. Unary ops (neg, abs)

## Future Phases

- Phase 5: Broadcasting
- Phase 6: Ufuncs framework
- Phase 7: Reductions (sum, mean, max)
- Phase 8: BLAS integration

## Testing Pattern

Every feature: do same operation in rumpy and numpy, use assert_eq():
```python
def test_feature():
    r = rumpy.some_op(...)
    n = np.some_op(...)
    assert_eq(r, n)
```

## Design Docs

See `designs/array-memory.md` for memory model principles.
