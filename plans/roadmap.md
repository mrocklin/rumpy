<!-- AI: NumPy API compatibility roadmap -->

# Rumpy Roadmap

Goal: Full NumPy API compatibility, building foundational pieces first.

## Current Status

**Done:**
- Core ndarray (shape, strides, views, Arc-based memory)
- 14 dtypes (float16/32/64, int16/32/64, uint8/16/32/64, bool, datetime64, complex128)
- Broadcasting for binary ops
- Integer/slice indexing
- Ufunc registry with NumPy-compatible type promotion
- Strided inner loops with contiguous fast path
- Basic ufuncs (arithmetic, trig, exp/log, floor/ceil, inverse trig)
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

## Phase D: Linear Algebra ✓

- [x] `matmul()`, `@` operator, `dot()`, `inner()`, `outer()`
- [x] `linalg.inv()`, `linalg.solve()`
- [x] `linalg.qr()`, `linalg.svd()`, `linalg.eigh()`
- [x] `linalg.norm()`, `linalg.det()`, `linalg.trace()`, `linalg.diag()`

## Phase E: Extended Dtypes (Partial)

- [x] `complex128`
- [x] `float16`, `int16`, `uint16`
- [ ] `complex64`
- [ ] String/unicode arrays
- [ ] Structured arrays (named fields)

## Phase F: Specialized Submodules (Partial)

- [x] `random` - Generator with random, integers, uniform, normal, exponential
- [x] `fft` - fft, ifft, fft2, ifft2, rfft, irfft, fftshift, ifftshift, fftfreq, rfftfreq
- [ ] I/O - `save()`, `load()`, `loadtxt()`, `savetxt()`

## Guiding Principles

1. **Foundational first** - Build primitives that other features depend on
2. **Test against NumPy** - Every feature validated with assert_eq(rumpy, numpy)
3. **Views over copies** - Prefer Arc-based sharing when possible
4. **Simple over optimized** - Correctness first, optimize hot paths later
