<!-- AI: Read this before starting work -->

# Current: API Compatibility Cleanup (Complete)

**Roadmap**: See `plans/roadmap.md` for overall plan.

## Status

**Complete**: 283 tests passing. API now matches NumPy patterns.

### API Changes Made

- [x] `shape` property returns tuple (was list)
- [x] `strides` property returns tuple (was list)
- [x] `zeros()`, `ones()`, `full()` accept int/tuple/list for shape
- [x] `reshape()` accepts varargs: `arr.reshape(3, 4)` or `arr.reshape((3, 4))`
- [x] `arange()` defaults to int64 (matches NumPy)
- [x] All tests updated to use NumPy-style calls

### Before/After

```python
# Before                          # After (now works!)
rp.zeros([10])                    rp.zeros(10)
rp.zeros([3, 4])                  rp.zeros((3, 4))
arr.reshape([3, 4])               arr.reshape(3, 4)
arr.shape == [3, 4]               arr.shape == (3, 4)
```

## Next: Phase D

Linear algebra: `matmul()`, `@` operator, `dot()`, `linalg.*`

## Known Limitations (Future Work)

- dtype accepts strings only (`"float64"`), not `np.float64`

## Build & Test

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
pytest tests/ -v
```
