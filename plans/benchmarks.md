# Benchmarks Plan

## Goal

Benchmark suite comparing rumpy vs numpy on realistic workloads. Each benchmark function takes `(xp, size)` where `xp` is either numpy or rumpy, enabling identical code paths.

## Current State

Benchmark suite in `benchmarks/` with 5 modules, 35 benchmarks total:
- `bench_ufuncs.py` - 8 benchmarks (chained element-wise)
- `bench_statistics.py` - 8 benchmarks (statistical computations)
- `bench_signal.py` - 7 benchmarks (FFT and signal processing)
- `bench_linalg.py` - 9 benchmarks (linear algebra workflows)
- `bench_simulation.py` - 8 benchmarks (Monte Carlo, physics, finance)

## Benchmark Results Summary

Run: `python benchmarks/run.py --sizes 10000`

### Passing Benchmarks (29/35)

All linalg, most ufuncs, most statistics, most signal, and most simulation benchmarks pass.

| Module | Passing | Failing |
|--------|---------|---------|
| bench_linalg | 9/9 | 0 |
| bench_ufuncs | 7/8 | GELU (numpy.float64 * array) |
| bench_statistics | 6/8 | bincount, percentile |
| bench_signal | 5/7 | lowpass (__setitem__), convolve |
| bench_simulation | 7/8 | heat eq (concatenate), random walk (choice) |

### Failing Benchmarks (6/35)

#### numpy.float64 * array returns numpy array
- **GELU**: `np.sqrt(2/pi) * x` - numpy.float64 takes precedence
- **Black-Scholes**: same issue (numpy scalar * rumpy array)
- Fix: Implement `__array_ufunc__` to tell numpy to defer to us

#### Missing slice assignment (`__setitem__`)
- **FFT lowpass filter**: `f[cutoff:-cutoff] = 0`

#### Missing functions
- **convolve 1D**: `rp.convolve` not implemented
- **histogram**: `rp.bincount` not implemented
- **percentile**: `rp.percentile` not implemented
- **random walk**: `Generator.choice` not implemented
- **heat equation**: `rp.concatenate` with Python lists not supported

## Recently Completed

1. ✅ **`rp.linalg` submodule** - solve, qr, svd, eigh, cholesky, inv, det, norm
2. ✅ **`rp.newaxis`** - added as None constant
3. ✅ **`rp.linalg.cholesky`** - Cholesky decomposition
4. ✅ **Scalar ufuncs** - rp.log(3.14), rp.sqrt(scalar), etc. now work
5. ✅ **None/newaxis in indexing** - `arr[:, None, :]` now works
6. ✅ **Scalar support in maximum/minimum** - `rp.maximum(arr, 1e-10)` now works

## Remaining API Gaps (Prioritized)

### Priority 1: Core Operations

1. **`__array_ufunc__` on PyRumpyArray**
   - Makes numpy defer arithmetic to rumpy when mixed
   - `np.sqrt(2/pi) * rp_array` should return rumpy array
   - Affects: GELU, Black-Scholes

### Priority 2: Array Operations

2. **Slice assignment (`__setitem__`)**
   - `arr[5:10] = 0` not supported
   - Affects: FFT filtering

3. **`rp.concatenate` with Python lists**
   - Should accept `[[scalar], array, [scalar]]`
   - Affects: heat equation

### Priority 3: Missing Functions

4. `Generator.choice(values, size=n)`
5. `rp.convolve(x, kernel, mode='same')`
6. `rp.bincount(x)`
7. `rp.percentile(x, [25, 50, 75])`

## Next Steps

1. Implement `__array_ufunc__` for numpy interop
2. Implement `__setitem__` for slice assignment
3. Address remaining missing functions

## Running Benchmarks

```bash
cd benchmarks
python run.py                              # Run all
python run.py --sizes 10000,100000         # Custom sizes
python run.py --module bench_ufuncs        # Specific module
python run.py --filter kalman              # Filter by name
```

## Performance Notes

Current speedups are ~0.01-0.07x (rumpy is 10-100x slower). Contributing factors:
1. Debug build (`maturin develop` without `--release`)
2. No SIMD vectorization
3. Python object allocation for intermediates

For realistic performance comparison:
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop --release
```
