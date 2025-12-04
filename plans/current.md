<!-- AI: Read this before starting work -->

# Current: Phase B - Array Manipulation (Complete)

**Roadmap**: See `plans/roadmap.md` for overall plan.

## Status

**Complete**: 24 new tests, 267 total passing.

- [x] `copy()` - explicit copy
- [x] `astype()` - dtype conversion
- [x] `concatenate()`, `stack()`, `vstack()`, `hstack()`
- [x] `split()`, `array_split()`
- [x] `squeeze()`, `expand_dims()`

## Implementation

**Rust** (`src/array/mod.rs`):
- `copy()`, `squeeze()`, `expand_dims()`, `astype()` methods on RumpyArray
- `concatenate()`, `stack()`, `split()`, `array_split()` functions

**Python** (`src/python/`):
- Methods on ndarray class: `copy()`, `squeeze()`, `astype()`
- Module functions: `expand_dims()`, `squeeze()`, `concatenate()`, `stack()`, `vstack()`, `hstack()`, `split()`, `array_split()`

## Next: Phase C

Reductions + sorting: `std()`, `var()`, `argmax()`, `argmin()`, `sort()`, `argsort()`, `unique()`

## Build & Test

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
pytest tests/ -v
```
