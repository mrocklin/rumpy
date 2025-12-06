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

### Passing Benchmarks (15/35)

| Benchmark | Speedup | Notes |
|-----------|---------|-------|
| softmax | 0.02x | |
| sigmoid | 0.07x | |
| normalize | 0.01x | |
| relu + scale + bias | 0.01x | |
| swish | 0.07x | |
| z-score 2D | 0.01x | |
| weighted mean | 0.01x | |
| covariance | 0.01x | |
| variance axis | 0.00x | |
| cumsum | 0.06x | |
| diff + cumsum | 0.01x | |
| FFT roundtrip | 0.01x | |
| FFT2 roundtrip | 0.01x | |
| power spectrum | 0.01x | |
| gradient (diff) | 0.00x | |
| second derivative | 0.00x | |
| Kalman predict | 0.00x | |
| matrix power | 0.00x | |
| Monte Carlo pi | 0.03x | |
| vector field magnitude | 0.01x | |
| finite difference laplacian | 0.03x | |

### Failing Benchmarks (20/35)

#### Module-level ufuncs need scalar support
- **log-softmax**: `rp.log(scalar)` fails - ufuncs don't accept Python floats
- **layer norm**: `rp.sqrt(scalar)` fails - same issue
- **GELU**: `scalar * array` returns numpy array (no `__rmul__`)
- **Black-Scholes**: `scalar * array` returns numpy array

#### Missing `rp.linalg` submodule
- **Kalman update**: needs `rp.linalg.solve`
- **least squares (SVD)**: needs `rp.linalg.svd`
- **QR solve**: needs `rp.linalg.qr`, `rp.linalg.solve`
- **Gram-Schmidt**: needs `rp.linalg.qr`
- **eigendecomposition**: needs `rp.linalg.eigh`
- **Cholesky solve**: needs `rp.linalg.cholesky`
- **matrix norm**: needs `rp.linalg.norm`

#### Missing `rp.newaxis`
- **pairwise distances**: `points[:, xp.newaxis, :]`
- **N-body acceleration**: `pos[xp.newaxis, :, :]`

#### Missing slice assignment (`__setitem__`)
- **FFT lowpass filter**: `f[cutoff:-cutoff] = 0`

#### Missing `rp.convolve`
- **convolve 1D**: needs `xp.convolve(x, kernel, mode='same')`

#### Missing `rp.concatenate` with mixed types
- **heat equation step**: `xp.concatenate([[u[0]], u_interior, [u[-1]]])`

#### Missing `Generator.choice`
- **random walk**: needs `rng.choice([-1, 1], size=size)`

#### Missing `rp.bincount`, `rp.percentile`
- **histogram**: needs `xp.bincount(x)`
- **percentile**: needs `xp.percentile(x, [25, 50, 75])`

## API Gaps (Prioritized)

### Priority 1: Core Operations
These affect many benchmarks and are fundamental to numpy compatibility.

1. **Scalar support in module-level ufuncs**
   - `rp.log(3.14)` should return `float`
   - `rp.sqrt(x.var())` should work when var() returns scalar
   - Affects: log-softmax, layer norm, others

2. **`rp.linalg` submodule**
   - Functions exist (`rp.solve`, `rp.qr`, etc.) but not under `linalg` namespace
   - Fix: Create `rp.linalg` submodule that re-exports existing functions
   - Affects: 7 linalg benchmarks

3. **`rp.newaxis`**
   - Should be `None` (same as numpy)
   - Used for broadcasting: `arr[:, np.newaxis]`
   - Affects: pairwise distances, N-body

4. **Scalar-first arithmetic (`__rmul__`, `__radd__`, etc.)**
   - `0.5 * rp_array` returns numpy array instead of rumpy array
   - Fix: Implement `__rmul__`, `__radd__`, `__rsub__`, `__rtruediv__`
   - Affects: GELU, Black-Scholes, any `scalar * array`

### Priority 2: Array Operations

5. **Slice assignment (`__setitem__`)**
   - `arr[5:10] = 0` not supported
   - Affects: FFT filtering, in-place updates

6. **`rp.concatenate` with Python lists**
   - Currently requires all rumpy arrays
   - Should accept `[[scalar], array, [scalar]]`
   - Affects: heat equation

### Priority 3: Missing Functions

7. **`Generator.choice(values, size=n)`**
   - Random selection from array of values
   - Affects: random walk

8. **`rp.convolve(x, kernel, mode='same')`**
   - 1D convolution
   - Affects: signal processing

9. **`rp.bincount(x)`**
   - Count occurrences of integers
   - Affects: histogram

10. **`rp.percentile(x, [25, 50, 75])`**
    - Compute percentiles
    - Affects: statistics

## Next Steps

1. Fix Priority 1 items (scalar ufuncs, linalg namespace, newaxis, __rmul__)
2. Run benchmarks again to get more passing
3. Address Priority 2 and 3 as needed
4. Once API complete, focus on performance optimization

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
