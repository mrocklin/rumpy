# Known Issues

## Current Limitations

### Tuple axis with second-order statistics (var, std, ptp)
**Impact**: Operations like variance, standard deviation, and peak-to-peak give incorrect results when using tuple of axes.

```python
n = np.arange(24).reshape(2, 3, 4)
r = rp.asarray(n)

# These give incorrect results:
rp.var(r, axis=(0, 2))  # Wrong
rp.std(r, axis=(0, 2))  # Wrong
rp.ptp(r, axis=(0, 2))  # Wrong
```

**Workaround**: Use single axis or reduce over all axes (axis=None).

**Operations that work correctly with tuple axis**:
sum, prod, mean, max, min, all, any, median, nansum, nanmean, nanmax, nanmin

## Fixed Issues

### 1. Negative axis on 3D+ arrays ✓
Axis normalization now handles negative indices correctly.
Tests: `test_reductions.py::TestStandardReductions::test_3d_axis`

### 2. Tuple of axes ✓
Reduction functions support tuple of axes.
Tests: `test_reductions.py::TestStandardReductions::test_tuple_axis`

### 3. Small integer dtypes ✓
int8, int16, uint16 are now supported.
Tests: `test_dtypes.py::TestSmallIntegerDtypes`

### 4. float16 numpy interop ✓
float16 arrays can be imported from numpy.
Tests: `test_dtypes.py::TestFloat16Interop`
