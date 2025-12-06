# Benchmarks Plan

## Goal

Benchmark suite comparing rumpy vs numpy on realistic workloads. Each benchmark function takes `(xp, size)` where `xp` is either numpy or rumpy, enabling identical code paths.

## Current State

Benchmark suite in `benchmarks/` with 5 modules, 40 benchmarks total:
- `bench_ufuncs.py` - 8 benchmarks (chained element-wise)
- `bench_statistics.py` - 8 benchmarks (statistical computations)
- `bench_signal.py` - 7 benchmarks (FFT and signal processing)
- `bench_linalg.py` - 9 benchmarks (linear algebra workflows)
- `bench_simulation.py` - 8 benchmarks (Monte Carlo, physics, finance)

## Benchmark Results Summary

Run: `python benchmarks/run.py --sizes 10000`

### All 40 Benchmarks Passing ✅

| Module | Passing |
|--------|---------|
| bench_linalg | 9/9 |
| bench_ufuncs | 8/8 |
| bench_statistics | 8/8 |
| bench_signal | 7/7 |
| bench_simulation | 8/8 |

## Recently Completed

1. ✅ **`rp.linalg` submodule** - solve, qr, svd, eigh, cholesky, inv, det, norm
2. ✅ **`rp.newaxis`** - added as None constant
3. ✅ **`rp.linalg.cholesky`** - Cholesky decomposition
4. ✅ **Scalar ufuncs** - rp.log(3.14), rp.sqrt(scalar), etc. now work
5. ✅ **None/newaxis in indexing** - `arr[:, None, :]` now works
6. ✅ **Scalar support in maximum/minimum** - `rp.maximum(arr, 1e-10)` now works
7. ✅ **`__array_ufunc__`** - numpy defers to rumpy for mixed operations
8. ✅ **`__setitem__`** - slice assignment (`arr[2:5] = 0`)
9. ✅ **Binary ufuncs** - `rp.add`, `rp.multiply`, `rp.subtract`, `rp.divide`, etc.

## Running Benchmarks

```bash
cd benchmarks
python run.py                              # Run all
python run.py --sizes 10000,100000         # Custom sizes
python run.py --module bench_ufuncs        # Specific module
python run.py --filter kalman              # Filter by name
```

## Performance Notes

**Release build results** (size=1M):

| Category | Speedup Range | Highlights |
|----------|---------------|------------|
| Ufuncs | 0.4-1.0x | sigmoid 1.04x, swish 0.87x |
| Statistics | 0.1-1.2x | cumsum 1.18x, diff 0.03x |
| Signal | 0.03-0.6x | FFT ~0.5x, diff needs work |
| Linalg | 0.2-0.9x | SVD 0.85x, norm 0.03x |
| Simulation | 0.1-1.4x | laplacian 1.35x |

**Wins** (>1x): cumsum, sigmoid, finite difference laplacian
**Needs work**: diff (0.03x), norm (0.03x), gradient (0.03x)

See `plans/performance.md` for detailed analysis and improvement plan.

Build with release:
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop --release
```
