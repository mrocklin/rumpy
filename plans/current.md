<!-- AI: Read this before starting work -->

# Current: API Compatibility

**Roadmap**: See `plans/roadmap.md` for overall plan.

## Status

**Complete**: 307 tests passing. Gufunc infrastructure and matmul implemented.

### Recent Work

- [x] **Gufunc infrastructure** (`src/ops/gufunc/`) - see `designs/gufuncs.md`
  - Signature parsing: `"(m,n),(n,p)->(m,p)"`
  - Dimension resolution with broadcasting
  - `GufuncKernel` trait for BLAS-ready extensibility
  - `gufunc_call()` loop over broadcast dimensions
- [x] **matmul**: `rp.matmul(a, b)` and `@` operator
  - 2D matrices, 1D vectors, batched (3D+)
  - Full numpy broadcast semantics
- [x] `__repr__` and `__str__` match NumPy format exactly
- [x] `shape` property returns tuple (was list)
- [x] `strides` property returns tuple (was list)
- [x] `zeros()`, `ones()`, `full()` accept int/tuple/list for shape
- [x] `reshape()` accepts varargs: `arr.reshape(3, 4)` or `arr.reshape((3, 4))`
- [x] `arange()` defaults to int64 (matches NumPy)

### repr/str Examples

```python
>>> rp.arange(12).reshape(3, 4)
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

>>> print(rp.arange(5))
[0 1 2 3 4]
```

## Next: More Gufuncs & Linear Algebra

Additional gufuncs using the infrastructure:
- `dot()` - inner product with different semantics than matmul
- `inner()` / `outer()` - inner and outer products
- `linalg.*` - determinant, trace, etc.

## Known Limitations (Future Work)

- dtype accepts strings only (`"float64"`), not `np.float64`
- `asarray()` doesn't infer bool dtype from Python bools

## Build & Test

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
pytest tests/ -v
```
