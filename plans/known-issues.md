# Known Issues

Gaps in numpy compatibility discovered during test redesign.
See `tests/test_known_issues.py` for xfail tests that document these.

## High Priority

### 1. Negative axis broken on 3D+ arrays
**Error**: `can't convert negative int to unsigned`

```python
n = np.arange(24).reshape(2, 3, 4)
r = rp.asarray(n)
rp.sum(r, axis=-1)  # ERROR
```

**Impact**: Any reduction with negative axis on 3D+ data fails.
**Location**: Likely in axis normalization code in Rust.

### 2. Tuple of axes not supported
**Error**: `'tuple' object cannot be interpreted as an integer`

```python
rp.sum(r, axis=(0, 2))  # ERROR
```

**Impact**: Can't reduce over multiple axes at once.
**Location**: Python bindings don't handle tuple axis parameter.

## Medium Priority

### 3. Missing small integer dtypes
**Unsupported**: int8, int16, uint16

```python
n = np.array([1, 2, 3], dtype=np.int8)
r = rp.asarray(n)  # ERROR: Unsupported dtype: |i1
```

**Impact**: Can't interop with numpy arrays using these dtypes.
**Note**: uint8 works, int32/int64/uint32/uint64 work.

### 4. float16 import from numpy
**Error**: `Unsupported dtype: <f2`

```python
n = np.array([1.0], dtype=np.float16)
r = rp.asarray(n)  # ERROR
```

**Note**: `rp.zeros(5, dtype="float16")` works - only numpy interop is broken.

## Workarounds

1. **Negative axis**: Convert to positive: `axis = axis % ndim`
2. **Tuple axis**: Chain reductions: `rp.sum(rp.sum(r, axis=2), axis=0)`
3. **Small dtypes**: Cast to int32/int64 before passing to rumpy
4. **float16**: Use float32 or create directly with rumpy

## Test Coverage Notes

The test redesign inadvertently avoided these issues by:
- Using only int32/int64/uint32/uint64 in INT_DTYPES
- Testing negative axis only on 1D/2D arrays
- Not testing tuple axis parameter
- Not testing numpy interop for float16

Future test additions should explicitly probe edge cases.
