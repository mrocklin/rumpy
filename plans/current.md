<!-- AI: Read this before starting work on rumpy -->

# Current Development Status

**Last updated**: Phase 1 complete

## What's Done

- [x] Project scaffolding (Cargo.toml, pyproject.toml, maturin)
- [x] Core types: DType, ArrayFlags, ArrayBuffer, RumpyArray
- [x] Basic constructors: zeros(), ones()
- [x] Python bindings with __array_interface__
- [x] Test infrastructure with assert_eq helper
- [x] All 22 tests passing

## Key Files

- `src/array/mod.rs` - RumpyArray struct and constructors
- `src/array/dtype.rs` - DType enum
- `src/python/pyarray.rs` - Python class wrapper
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

## Next Phase: Views and Slicing

Priority order:
1. `view_with()` - core view creation
2. `__getitem__` for basic slicing (arr[1:5], arr[::2])
3. `reshape()` - return view when contiguous
4. `transpose()` / `.T` property

## Future Phases

- Phase 3: Element access, arange, linspace
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
