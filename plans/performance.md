# Performance Plan

## Current State (Release Build)

Benchmarks at size=1M with release build (2025-12-06):

| Category | Best | Worst | Typical |
|----------|------|-------|---------|
| Ufuncs (chained) | 1.04x (swish) | 0.43x (relu) | 0.6-1.0x |
| Statistics | 1.11x (diff+cumsum) | 0.34x (weighted mean) | 0.4-1.0x |
| Signal/FFT | 1.30x (second deriv) | 0.41x (FFT lowpass) | 0.4-1.1x |
| Linalg | 0.66x (eigh) | 0.07x (norm) | 0.1-0.6x |
| Simulation | 1.44x (laplacian) | 0.12x (vector mag) | 0.5-0.6x |

**Wins** (>1x): laplacian (1.44x), second deriv (1.30x), diff+cumsum (1.11x), diff (1.08x), cumsum (1.04x), swish (1.04x)

## Completed Optimizations

- [x] **diff**: Rewrote to use registry Sub loop (0.03x → 1.0x)
- [x] **tanh/sinh/cosh**: Added to registry for f32, f64, f16
- [x] **var (full array)**: Contiguous f64 fast path with two-pass algorithm (0.23x → 0.86x)
- [x] **moment/skew/kurtosis**: Added higher-order statistics functions
- [x] **var_axis/moment_axis**: Rewritten to compose vectorized ops (0.17x → 0.45x)
- [x] **matrix norm**: Use vectorized square + sum (0.03x → 0.07x)
- [x] **z-score 2D**: Benefits from var_axis optimization (0.34x → 0.95x)

## Analysis

### Where rumpy is competitive (0.8x-1.5x)

1. **Finite difference** (laplacian 1.44x, second deriv 1.30x): Slicing creates views, registry loops
2. **Streaming ops** (diff+cumsum 1.11x, diff 1.08x, cumsum 1.04x): Single-pass operations with registry loops
3. **Simple ufuncs** (swish 1.04x, sigmoid 0.96x, layer norm 0.96x, GELU 0.89x): Registry loops enable LLVM auto-vectorization
4. **Axis stats** (z-score 2D 0.95x): Vectorized moment_axis composition

### Where rumpy is slow (0.1x-0.5x)

1. **Reduction chains** (relu+scale+bias 0.43x, covariance 0.40x):
   - Each op allocates a new array
   - No fusion of elementwise operations

2. **Linalg** (norm 0.07x, matrix power 0.14x, Kalman 0.16-0.18x):
   - Norm still allocates intermediate for x*x
   - Copying between rumpy and faer formats
   - Some ops call many small matrix operations

3. **Vector field ops** (vector magnitude 0.12x):
   - sqrt(x² + y² + z²) pattern has allocation overhead

4. **FFT** (~0.44x): RustFFT is slower than FFTW/numpy

## Recommended Next Steps

### High Impact

1. **Optimize vector magnitude pattern** (currently 0.12x)
   - sqrt(x² + y² + z²) creates 4 intermediate arrays
   - Could add hypot3(x, y, z) ufunc or fused loop

2. **Fused norm** (currently 0.07x)
   - Still allocates intermediate for x*x before sum
   - Could add registry-based sum-of-squares reduction

### Moderate Impact

3. **Reduce allocation in reduction chains** (relu 0.43x, covariance 0.40x)
   - Options: scalar_op fast path, expression templates, or fused ops

4. **Linalg overhead** (Kalman 0.16-0.18x, matrix power 0.14x)
   - Profile to understand if it's conversion or algo overhead

### Lower Priority

5. **FFT** (~0.44x): RustFFT vs FFTW is fundamental library choice
6. **Expression templates**: Significant effort, would help allocation-heavy patterns

## Build Notes

Always benchmark with release:
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop --release
```
