# Testing Philosophy

Comprehensive numpy-parity testing for rumpy.

## Core Principles

1. **NumPy is truth** - Every test compares rumpy against numpy. Never test rumpy in isolation.
2. **Parametrize over ufuncs** - Group operations by domain, parametrize over them.
3. **Tiered dtype coverage** - Representative dtypes for wrappers, full for custom code.
4. **Edge cases are first-class** - Empty arrays, NaN, views get explicit tests.
5. **Minimal essential shapes** - Three core shapes catch most bugs.

## Dtype Tiers (in conftest.py)

```python
FLOAT_DTYPES = ["float32", "float64"]
INT_DTYPES = ["int32", "int64"]
UINT_DTYPES = ["uint32", "uint64"]
CORE_DTYPES = ["float64", "float32", "int64", "bool"]  # Simple wrappers
NUMERIC_DTYPES = FLOAT_DTYPES + INT_DTYPES + UINT_DTYPES  # Custom logic
BITWISE_DTYPES = INT_DTYPES + UINT_DTYPES + ["bool"]
```

## Shape Constants (in conftest.py)

```python
CORE_SHAPES = [(5,), (3, 4), (2, 3, 4)]
SHAPES_EMPTY = [(0,), (0, 5), (5, 0)]
SHAPES_BROADCAST = [((3, 1), (1, 4)), ((1,), (5,))]
```

## Helper Functions (in helpers.py)

```python
assert_eq(r, n, rtol=1e-7, atol=1e-14)  # Compare rumpy vs numpy
make_numpy(shape, dtype)                 # Generate test array
make_pair(shape, dtype)                  # Returns (rumpy, numpy) pair
make_positive_pair(shape, dtype)         # For sqrt, log, etc.
```

## Test Pattern: Parametrize Over Ufuncs

Group operations by input domain, then parametrize:

```python
UNRESTRICTED_UFUNCS = ["exp", "sin", "cos", "abs", "sign", "square"]
POSITIVE_UFUNCS = ["sqrt", "log", "log10", "log2"]
BOUNDED_UFUNCS = ["arcsin", "arccos", "arctanh"]  # [-1, 1]

class TestUnrestrictedUfuncs:
    @pytest.mark.parametrize("ufunc", UNRESTRICTED_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n = np.array([-2, -1, 0, 1, 2], dtype=dtype)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r), np_fn(n))
```

## Test Pattern: Reductions

```python
STANDARD_REDUCTIONS = ["sum", "prod", "max", "min"]

class TestStandardReductions:
    @pytest.mark.parametrize("reduction", STANDARD_REDUCTIONS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, reduction, dtype):
        n = np.array([1, 2, 3, 4, 5], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(getattr(r, reduction)(), getattr(n, reduction)())
```

## Known Rumpy Limitations (work around in tests)

- `rp.diag` only works with float arrays, not int
- Negative axis doesn't work reliably for 3D+ arrays
- `count_nonzero` doesn't support `axis` parameter
- `rp.mod`, `rp.true_divide`, unary `+` not implemented
- float32 arctanh needs rtol=1e-5 (lower precision)
