# Performance Plan

## Current State (Release Build, Post-P0 Fixes)

Benchmarks at size=1M with release build:

| Category | Best | Worst | Typical |
|----------|------|-------|---------|
| Ufuncs (chained) | 0.95x (swish) | 0.41x (relu) | 0.5-0.9x |
| Statistics | 1.15x (diff+cumsum) | 0.18x (variance) | 0.3-1.0x |
| Signal/FFT | 1.16x (second deriv) | 0.42x (FFT) | 0.4-1.0x |
| Linalg | 0.63x (eigh) | 0.03x (norm) | 0.2-0.5x |
| Simulation | 1.34x (laplacian) | 0.13x (vector mag) | 0.3-0.6x |

**Wins** (>1x): cumsum, diff, second derivative, finite difference laplacian

## Completed Optimizations

- [x] **diff**: Rewrote to use registry Sub loop (0.03x → 1.0x)
- [x] **tanh/sinh/cosh**: Added to registry for f32, f64, f16

## Analysis

### Where rumpy is competitive (0.8x-1.3x)

1. **Simple ufunc chains** (sigmoid, swish): Registry loops enable LLVM auto-vectorization
2. **cumsum, diff**: Single-pass operations with registry loops
3. **Finite difference**: Slicing creates views, minimal overhead

### Where rumpy is slow (0.1x-0.5x)

1. **Reduction chains** (normalize, relu+scale+bias):
   - Each op allocates a new array
   - No fusion of elementwise operations
   - Example: `(x - mean) / std` = 4 allocations + 4 passes

2. **Axis reductions** (var, std):
   - Two passes: mean then variance
   - Per-element indexing overhead

3. **Matrix operations** (matmul at scale):
   - Copying between rumpy and faer formats

4. **Broadcasting**:
   - Strided iteration has branch overhead
   - No contiguous fast path for broadcast ops

## Remaining P0 Work

- [ ] Welford's algorithm for `var`/`std` (single-pass)
- [ ] Fast path in `scalar_op` to avoid allocation

## P1 Improvements (Moderate Effort)

- [ ] Fused multiply-add for common patterns (`a * b + c`)
- [ ] Optimize broadcast binary ops (contiguous + broadcast fast path)
- [ ] Cache-aware blocking for reductions

## P2 Improvements (Significant Effort)

- [ ] Expression templates / lazy evaluation
- [ ] SIMD intrinsics (`std::simd`)
- [ ] Parallel iteration (rayon)

## Build Notes

Always benchmark with release:
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop --release
```
