<!-- AI: Read this before starting work -->

# Current: Fancy Indexing

**Roadmap**: See `plans/roadmap.md` for overall plan.

## Goal

Enable `arr[[0, 2, 4]]` - indexing with integer arrays.

## Status

- [ ] Integer array indexing (1D index into 1D array)
- [ ] Multi-dimensional fancy indexing
- [ ] Combined fancy + slice indexing

## Implementation

**Rust** (`src/array/mod.rs`):
- `select_by_indices(&self, indices: &RumpyArray) -> RumpyArray`
- Indices array contains integer positions to select
- Returns new array with selected elements

**Python** (`src/python/pyarray.rs`):
- Extend `__getitem__` to detect integer array argument
- Call `select_by_indices`

## NumPy Behavior

```python
arr = np.arange(10)
arr[[0, 2, 4]]        # [0, 2, 4] - select specific indices
arr[[0, 0, 1, 1]]     # [0, 0, 1, 1] - duplicates allowed

# 2D
arr = np.arange(12).reshape(3, 4)
arr[[0, 2], [1, 3]]   # [1, 11] - paired indexing
arr[[0, 2]]           # rows 0 and 2
```

## Build & Test

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
pytest tests/ -v
```
