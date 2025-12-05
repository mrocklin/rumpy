<!-- AI: Read this before starting work -->

# Current: API Compatibility

**Roadmap**: See `plans/roadmap.md` for overall plan.

## Status

**Complete**: 356 tests passing. Linalg: matmul, dot, inner, outer, solve, trace, det, norm, qr, svd, inv, eigh, diag.

### Recent Work

- [x] **Gufunc infrastructure** (`src/ops/gufunc/`) - see `designs/gufuncs.md`
  - Signature parsing: `"(m,n),(n,p)->(m,p)"`
  - Dimension resolution with broadcasting
  - `GufuncKernel` trait for BLAS-ready extensibility
  - `gufunc_call()` loop over broadcast dimensions
- [x] **matmul**: `rp.matmul(a, b)` and `@` operator
  - 2D matrices, 1D vectors, batched (3D+)
  - Full numpy broadcast semantics
  - Uses faer for optimized matrix multiplication
- [x] **dot**: `rp.dot(a, b)` - numpy-compatible dot product
- [x] **inner**: `rp.inner(a, b)` - inner product with gufunc broadcasting
- [x] **outer**: `rp.outer(a, b)` - outer product (flattens inputs)
- [x] **solve**: `rp.solve(A, b)` - solve linear system Ax=b via LU decomposition
- [x] **trace**: `rp.trace(A)` - sum of diagonal elements
- [x] **det**: `rp.det(A)` - determinant via LU decomposition
- [x] **norm**: `rp.norm(A)` - Frobenius norm (default)
- [x] **qr**: `rp.qr(A)` - QR decomposition returning (Q, R)
- [x] **svd**: `rp.svd(A)` - SVD decomposition returning (U, S, Vt)
- [x] **inv**: `rp.inv(A)` - matrix inverse
- [x] **eigh**: `rp.eigh(A)` - symmetric eigendecomposition (w, V)
- [x] **diag**: `rp.diag(a)` - extract or create diagonal
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

## Next

- `eig`, `eigvals` - eigenvalue decomposition
- More norm types (1, 2, inf, nuclear)

## Known Limitations (Future Work)

- dtype accepts strings only (`"float64"`), not `np.float64`
- `asarray()` doesn't infer bool dtype from Python bools

## Build & Test

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
pytest tests/ -v
```
