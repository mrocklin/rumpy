<!-- AI: Read this before starting work -->

# Current: Phase C - Reductions + Sorting (Complete)

**Roadmap**: See `plans/roadmap.md` for overall plan.

## Status

**Complete**: 16 new tests, 283 total passing.

- [x] `std()`, `var()` - with axis support
- [x] `argmax()`, `argmin()` - flattened index
- [x] `sort()`, `argsort()` - flattened sorting
- [x] `unique()` - deduplicated values

## Implementation

**Rust** (`src/ops/mod.rs`):
- `var()`, `var_axis()`, `std()`, `std_axis()` - statistical reductions
- `argmax()`, `argmin()` - index of extrema
- `sort()`, `argsort()`, `unique()` - sorting operations

**Python**:
- Methods on ndarray: `var()`, `std()`, `argmax()`, `argmin()`
- Module functions: `sort()`, `argsort()`, `unique()`

## Next: Phase D

Linear algebra: `matmul()`, `@` operator, `dot()`, `linalg.*`

## Build & Test

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
pytest tests/ -v
```
