<!-- AI: Read this before starting work -->

# Phase 12: Comparison & Boolean Indexing

**Goal**: Enable `arr[arr > 5]` patterns and `where()` conditional selection.

## Status

- [x] **12a**: Comparison ops (`>`, `<`, `==`, `!=`, `>=`, `<=`)
- [x] **12b**: Boolean indexing (`arr[mask]`)
- [x] **12c**: `where(cond, x, y)`

**Completed**: 26 new tests, 230 total tests passing.

## 12a: Comparison Operations

**Rust** (`src/ops/mod.rs`):
- `compare_arrays(a, b, op) -> RumpyArray` returning bool dtype
- Use existing `map_binary` pattern with broadcasting

**Python** (`src/python/pyarray.rs`):
- `__gt__`, `__lt__`, `__eq__`, `__ne__`, `__ge__`, `__le__`
- Support array vs array and array vs scalar

## 12b: Boolean Indexing

**Rust** (`src/array/mod.rs`):
- `select_by_mask(&self, mask: &RumpyArray) -> RumpyArray`

**Python**: Extend `__getitem__` to accept bool array

## 12c: where()

`where(condition, x, y)` - broadcast all three, return x where true, y where false.

## Build & Test

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
pytest tests/ -v
```
