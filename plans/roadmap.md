<!-- AI: NumPy API compatibility roadmap -->

# Rumpy Roadmap

Goal: Full NumPy API compatibility, building foundational pieces first.

## Current Status

**Done:**
- Core ndarray (shape, strides, views, Arc-based memory)
- 9 dtypes (float32/64, int32/64, uint8/32/64, bool, datetime64)
- Broadcasting for binary ops
- Integer/slice indexing
- Basic ufuncs (arithmetic, trig, exp/log)
- Basic reductions (sum, prod, min, max, mean)
- Comparison ops (>, <, ==, !=, >=, <=)
- Boolean indexing (arr[mask])
- where(cond, x, y)

## Phase A: Complete Indexing ✓

- [x] Comparison ops
- [x] Boolean indexing
- [x] Fancy indexing - `arr[[0, 2, 4]]`, `arr[idx_array]`

## Phase B: Array Manipulation ✓

- [x] `copy()` - explicit copy
- [x] `astype()` - dtype conversion
- [x] `concatenate()`, `stack()`, `vstack()`, `hstack()`
- [x] `split()`, `array_split()`
- [x] `squeeze()`, `expand_dims()`

## Phase C: More Reductions + Sorting ✓

- [x] `std()`, `var()` - statistical essentials
- [x] `argmax()`, `argmin()` - index of extrema
- [x] `sort()`, `argsort()` - sorting
- [x] `unique()` - deduplicated values

## Phase D: Linear Algebra

- [ ] `matmul()`, `@` operator, `dot()`
- [ ] `linalg.inv()`, `linalg.solve()`
- [ ] `linalg.eig()`, `linalg.svd()`
- [ ] `linalg.norm()`

## Phase E: Extended Dtypes

- [ ] `complex64`, `complex128`
- [ ] String/unicode arrays
- [ ] Structured arrays (named fields)

## Phase F: Specialized Submodules

- [ ] `random` - random number generation
- [ ] `fft` - Fourier transforms
- [ ] I/O - `save()`, `load()`, `loadtxt()`, `savetxt()`

## Guiding Principles

1. **Foundational first** - Build primitives that other features depend on
2. **Test against NumPy** - Every feature validated with assert_eq(rumpy, numpy)
3. **Views over copies** - Prefer Arc-based sharing when possible
4. **Simple over optimized** - Correctness first, optimize hot paths later
