<!-- AI: Read this before starting work on rumpy -->

# Current Development Status

**Last updated**: Phase 10 complete

## What's Done

- [x] Project scaffolding (Cargo.toml, pyproject.toml, maturin)
- [x] Core types: DType, ArrayFlags, ArrayBuffer, RumpyArray
- [x] Basic constructors: zeros(), ones(), arange()
- [x] More constructors: linspace(), eye(), full()
- [x] asarray() with __array_interface__ and list support
- [x] Python bindings with __array_interface__
- [x] Test infrastructure with assert_eq helper
- [x] Views: view_with(), slicing, reshape, transpose
- [x] Integer indexing: arr[i], arr[i, j] returns scalar
- [x] Binary ops: add, sub, mul, div with broadcasting
- [x] Scalar ops: arr + 5, 5 + arr, etc.
- [x] Unary ops: neg, abs
- [x] Broadcasting: broadcast_shapes(), broadcast_to()
- [x] Ufunc machinery: map_unary, map_binary, reduce_all, reduce_axis
- [x] Reductions: sum, prod, min, max, mean (with optional axis)
- [x] Math ufuncs: sqrt, exp, log, sin, cos, tan
- [x] 171 tests passing

## Key Files

- `src/array/mod.rs` - RumpyArray struct, constructors, views, arange
- `src/array/dtype.rs` - DType enum
- `src/ops/mod.rs` - Binary and unary operations
- `src/python/pyarray.rs` - Python class with __getitem__, __add__, etc.
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

## Next Phase: BLAS integration

Priority order:
1. matmul / @ operator
2. dot product

## Future Phases

- Phase 12: Comparison ops (>, <, ==, etc.)
- Phase 13: More ufuncs (log2, log10, arcsin, arccos, etc.)
- Phase 14: copy() method, astype()

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
