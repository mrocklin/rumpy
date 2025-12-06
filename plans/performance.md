# Performance Plan

## Current State (Release Build)

Benchmarks at size=1M with release build show rumpy at **0.1x to 1.3x** numpy speed:

| Category | Best | Worst | Typical |
|----------|------|-------|---------|
| Ufuncs (chained) | 1.04x (sigmoid) | 0.43x (relu) | 0.5-0.8x |
| Statistics | 1.18x (cumsum) | 0.12x (diff+cumsum) | 0.2-0.3x |
| Signal/FFT | 1.18x (lowpass) | 0.03x (diff) | 0.4-0.5x |
| Linalg | 0.85x (SVD) | 0.03x (norm) | 0.2-0.5x |
| Simulation | 1.35x (laplacian) | 0.14x (vector mag) | 0.3-0.6x |

**Wins** (>1x): cumsum, sigmoid, swish, finite difference laplacian, FFT roundtrip (small sizes)

**Losses** (0.1x or worse): diff, gradient, matrix norm, relu+scale+bias, variance

## Analysis

### Where rumpy is competitive (0.8x-1.3x)

1. **Simple ufunc chains** (sigmoid, swish): Registry loops enable LLVM auto-vectorization
2. **cumsum**: Single-pass accumulation, simple loop
3. **Finite difference**: Slicing creates views, minimal overhead
4. **FFT at small sizes**: RustFFT competitive with numpy's pocketfft

### Where rumpy is slow (0.1x-0.5x)

1. **Reduction chains** (normalize, relu+scale+bias):
   - Each op allocates a new array
   - No fusion of elementwise operations
   - Example: `(x - mean) / std` = 4 allocations + 4 passes

2. **diff operation** (0.01-0.03x):
   - Uses `get_element` per element instead of strided loop
   - No fast path in registry

3. **Axis reductions** (var, std, sum_axis):
   - Per-element indexing with `byte_offset_for`
   - No SIMD, no blocking for cache

4. **Matrix operations** (matmul at scale):
   - Naive triple loop or faer overhead
   - Copying between rumpy and faer formats

5. **Broadcasting creates views but still iterates element-by-element**:
   - Strided iteration has branch overhead
   - No contiguous fast path for broadcast ops

## Priority Improvements

### P0: Low-hanging fruit (1-2 days each)

1. **Add fast paths for common operations**
   - `diff`: Register strided loop like other ufuncs
   - `var`/`std`: Fuse mean computation, use Welford's algorithm
   - Currently: 2 passes (mean, then variance)
   - Better: 1 pass Welford

2. **Avoid allocation in scalar operations**
   - `x - x.mean()` creates scalar array then broadcasts
   - Optimize: detect scalar, use in-place subtraction

3. **Add tanh/sinh/cosh to registry**
   - Currently falling back to trait dispatch
   - Simple addition to `register_float_unary!` macro

### P1: Moderate effort (1 week each)

4. **Fused multiply-add for common patterns**
   - `a * b + c` is very common (GELU, layer norm, etc.)
   - Add as compound operation in registry
   - LLVM can often emit FMA instructions

5. **Optimize broadcast binary ops**
   - Current: Always uses strided loop even when one input is contiguous
   - Better: Detect (contiguous, broadcast) pattern, use specialized loop

6. **Cache-aware blocking for reductions**
   - Current: Single loop over all elements
   - Better: Process in cache-line-sized chunks

### P2: Significant effort (2+ weeks)

7. **Expression templates / lazy evaluation**
   - Fuse `(x - mean) / std` into single pass
   - Requires new Array wrapper type
   - Major architectural change

8. **SIMD intrinsics**
   - Current: Rely on LLVM auto-vectorization
   - Better: Use `std::simd` or `packed_simd`
   - Most benefit for transcendentals (exp, log, sin)

9. **Parallel iteration**
   - Use rayon for large arrays
   - Threshold ~10K elements
   - Requires thread-safe buffer handling

10. **GPU offload (future)**
    - Use wgpu or cuda for very large arrays
    - Threshold ~100K elements

## Quick Wins Checklist

- [ ] Register `diff` in registry.rs with strided loop
- [ ] Register `tanh`, `sinh`, `cosh` in registry.rs
- [ ] Add `Maximum`, `Minimum` to binary registry
- [ ] Welford's algorithm for `var`/`std`
- [ ] Fast path in `scalar_op` to avoid allocation

## Measurement Plan

1. Profile individual operations with `cargo flamegraph`
2. Compare assembly output for contiguous vs strided paths
3. Benchmark at multiple sizes (1K, 10K, 100K, 1M) to see scaling

## Build Notes

Always benchmark with release:
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop --release
```

Debug builds are 10-100x slower due to bounds checks and no optimization.
