<!-- AI: Phase 12 implementation details -->

# Phase 12: Comparison Operations & Boolean Indexing

**Design doc**: None needed (straightforward extension of existing patterns)

## Goal

Enable filtering patterns like `arr[arr > 5]` and conditional selection via `where()`.

## Phase 12a: Comparison Operations

**Rust side** (`src/ops/mod.rs`):
- Add `ComparisonOp` enum: `Gt, Lt, Eq, Ne, Ge, Le`
- Add `compare_arrays(a, b, op) -> RumpyArray` using `map_binary` pattern
- Returns bool-dtype array

**Python side** (`src/python/pyarray.rs`):
- `__gt__`, `__lt__`, `__eq__`, `__ne__`, `__ge__`, `__le__`
- Each calls `compare_arrays` with appropriate op
- Support both array and scalar operands

**Tests** (`tests/test_comparison.py`):
```python
def test_gt_scalar():
    r = rumpy.arange(5) > 2
    n = np.arange(5) > 2
    assert_eq(r, n)  # [False, False, False, True, True]

def test_gt_array():
    a = rumpy.arange(5)
    b = rumpy.full(5, 2)
    assert_eq(a > b, np.arange(5) > np.full(5, 2))
```

## Phase 12b: Boolean Indexing

**Rust side** (`src/array/mod.rs`):
- `select_by_mask(&self, mask: &RumpyArray) -> RumpyArray`
- Mask must be bool dtype, same shape (or broadcastable)
- Returns 1D array of selected elements

**Python side** (`src/python/pyarray.rs`):
- Extend `__getitem__` to detect bool array argument
- Call `select_by_mask`

**Tests**:
```python
def test_bool_index():
    arr = rumpy.arange(10)
    mask = arr > 5
    r = arr[mask]
    n = np.arange(10)[np.arange(10) > 5]
    assert_eq(r, n)  # [6, 7, 8, 9]
```

## Phase 12c: where()

**Signature**: `where(condition, x, y) -> array`

- All three broadcast together
- Returns `x` where condition is true, `y` otherwise

**Implementation**: Use existing broadcast + element-wise iteration.

**Tests**:
```python
def test_where():
    cond = rumpy.arange(5) > 2
    r = rumpy.where(cond, 1, 0)
    n = np.where(np.arange(5) > 2, 1, 0)
    assert_eq(r, n)  # [0, 0, 0, 1, 1]
```

## Order of Implementation

1. Comparison ops (12a) - foundation for everything else
2. Boolean indexing (12b) - uses comparison results
3. where() (12c) - alternative conditional pattern

## Notes

- Bool dtype already exists and works
- Broadcasting already works
- Main work is wiring up comparisons and extending __getitem__
