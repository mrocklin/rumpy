<!-- AI: Read this before starting work on rumpy -->

# Current Development Status

**Last updated**: Phase 2 complete

## What's Done

- [x] Project scaffolding (Cargo.toml, pyproject.toml, maturin)
- [x] Core types: DType, ArrayFlags, ArrayBuffer, RumpyArray
- [x] Basic constructors: zeros(), ones()
- [x] Python bindings with __array_interface__
- [x] Test infrastructure with assert_eq helper
- [x] Views: view_with(), slicing, reshape, transpose
- [x] All 40 tests passing

## Key Files

- `src/array/mod.rs` - RumpyArray struct, constructors, views, slicing
- `src/array/dtype.rs` - DType enum
- `src/python/pyarray.rs` - Python class with __getitem__, reshape, T
- `tests/helpers.py` - assert_eq for comparing against numpy
- `tests/test_views.py` - slicing, reshape, transpose tests

## Building

```bash
# Create venv and install
uv venv && source .venv/bin/activate
uv pip install pytest numpy hypothesis
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop

# Run tests
pytest tests/ -v
```

## Next Phase: Element Access and Sequence Generation

Priority order:
1. `arange()` - generate sequences
2. Single element access `arr[i, j]` (integer indexing)
3. `linspace()` - evenly spaced values

## Future Phases

- Phase 4: Binary ops (add, mul) - same shape first
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
