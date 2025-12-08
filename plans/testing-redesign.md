# Testing Framework Redesign

**Goal**: Comprehensive numpy-parity testing with systematic dtype × shape × operation coverage.

## Current State

**2277 tests passing** across 21 test files. Phase 5 cleanup complete.

### New-Style Tests (DONE)
| File | Tests | Coverage |
|------|-------|----------|
| `test_creation.py` | 151 | zeros, ones, arange, eye, linspace, etc. |
| `test_unary.py` | 147 | sqrt, exp, log, sin, cos, etc. (parametrized by domain) |
| `test_binary.py` | 160 | add, mul, div, comparisons, broadcasting |
| `test_reductions.py` | 136 | sum, mean, argmax, nan*, cumsum, cumprod |
| `test_shape.py` | 159 | reshape, transpose, stack, split, flip, etc. |
| `test_indexing.py` | 136 | integer, slice, boolean, fancy indexing |
| `test_sorting.py` | 109 | sort, argsort, unique, partition, lexsort |
| `test_linalg.py` | 99 | matmul, dot, solve, inv, det, qr, svd, eig |
| `test_fft.py` | 363 | fft, ifft, fft2, rfft, fftfreq, fftshift |
| `test_bitwise.py` | 93 | &, |, ^, ~, <<, >> |
| `test_logical.py` | 84 | logical_and, logical_or, logical_not, logical_xor |

### Infrastructure
- `conftest.py` - Dtype tiers (FLOAT_DTYPES, NUMERIC_DTYPES, BITWISE_DTYPES), shape constants (CORE_SHAPES), fixtures
- `helpers.py` - assert_eq, make_pair, make_positive_pair, make_numpy

---

## Phase 5: Cleanup (DONE)

### 5A: Delete Superseded Old-Style Files
These files are now fully covered by new-style tests:

```bash
# Covered by test_unary.py
rm tests/test_ufuncs.py

# Covered by test_linalg.py
rm tests/test_matmul.py tests/test_dot.py tests/test_inner_outer.py tests/test_solve.py

# Covered by test_creation.py
rm tests/test_arange.py tests/test_constructors.py tests/test_asarray.py

# Covered by test_binary.py
rm tests/test_binary_math.py tests/test_comparison.py
```

### 5B: Rewrite Remaining Standalone Tests
These have unique content not yet in new-style format:

| Old File | Action | Notes |
|----------|--------|-------|
| `test_statistics.py` | REWRITE | median, cov, corrcoef, histogram, ptp, average |
| `test_nan.py` | MERGE into test_reductions.py | NaN-aware ops already partially there |
| `test_random.py` | REWRITE | Generator, distributions |
| `test_set.py` | REWRITE | isin, unique, intersect1d, union1d, setdiff1d |
| `test_numerical.py` | REWRITE | gradient, trapezoid, interp, correlate |
| `test_poly.py` | REWRITE | polyval, polyfit, polyder, polyint, roots |

### 5C: Consolidate Cross-Cutting Tests

| Old Files | Action | Notes |
|-----------|--------|-------|
| `test_views.py` + `test_broadcast.py` | MERGE | Cross-cutting concerns, merge into existing or create test_views.py |
| `test_dtypes.py` + `test_dtype_interactions.py` | CONSOLIDATE | One comprehensive dtype test file |

### 5D: Review and Clean
| File | Action | Notes |
|------|--------|-------|
| `test_array_methods.py` | REVIEW | Check for gaps not covered elsewhere |
| `test_new_functions.py` | DELETE if redundant | Review content first |
| `test_ops.py` | DELETE if redundant | Review content first |
| `test_repr.py` | KEEP | Representation tests are unique |
| `test_elision.py` | KEEP | Optimization tests are unique |

---

## Key Patterns (see designs/testing.md)

### Parametrize over ufuncs by domain
```python
UNRESTRICTED_UFUNCS = ["exp", "sin", "cos", "abs"]
POSITIVE_UFUNCS = ["sqrt", "log"]  # need x > 0

class TestPositiveUfuncs:
    @pytest.mark.parametrize("ufunc", POSITIVE_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n = np.array([0.5, 1, 2, 4], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(getattr(rp, ufunc)(r), getattr(np, ufunc)(n))
```

### Use helpers
```python
from helpers import assert_eq, make_pair, make_positive_pair
from conftest import CORE_SHAPES, FLOAT_DTYPES, NUMERIC_DTYPES

r, n = make_pair((3, 4), "float64")  # matching rumpy/numpy arrays
assert_eq(rp.sum(r), np.sum(n))
```

### Known rumpy limitations (work around in tests)
- `rp.diag` only works with float arrays, not int
- Negative axis doesn't work reliably for 3D+ arrays
- `count_nonzero` doesn't support `axis` parameter
- float32 arctanh needs rtol=1e-5

---

## Build & Test
```bash
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
pytest tests/ -v
```

## File Inventory After Cleanup

Target structure after Phase 5:
```
tests/
  conftest.py              # Fixtures, dtype/shape constants
  helpers.py               # assert_eq, make_pair utilities

  # Core operations (new-style, parametrized)
  test_creation.py         # Array creation
  test_unary.py            # Unary ufuncs
  test_binary.py           # Binary ops + comparisons
  test_reductions.py       # Reductions + NaN-aware
  test_shape.py            # Shape manipulation
  test_indexing.py         # Indexing operations
  test_sorting.py          # Sorting operations
  test_linalg.py           # Linear algebra
  test_fft.py              # FFT operations
  test_bitwise.py          # Bitwise operations
  test_logical.py          # Logical operations

  # Specialized (to rewrite)
  test_statistics.py       # Statistical functions
  test_random.py           # Random module
  test_set.py              # Set operations
  test_numerical.py        # Numerical operations
  test_poly.py             # Polynomial operations

  # Cross-cutting
  test_dtypes.py           # Dtype system
  test_views.py            # Views and broadcasting

  # Unique/keep
  test_repr.py             # Representation
  test_elision.py          # Optimization tests
```
