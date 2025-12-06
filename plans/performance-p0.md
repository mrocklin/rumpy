# Performance P0: Quick Wins

Parent: `plans/performance.md`

## Goal

Fix the worst performers (0.03x) and add missing registry entries. Target: get all benchmarks to at least 0.3x numpy speed.

## Completed

### 1. ✅ Optimize `diff` with registry Sub loop
**Before**: 0.03x numpy speed (used `get_element()` per element)
**After**: 1.0x numpy speed
**Fix**: Use registry's BinaryOp::Sub loop for contiguous 1D, strided loop for N-D

### 2. ✅ Add tanh/sinh/cosh to registry
**Fix**: Added to `register_float_unary!` macro and f16 unary loops in `registry.rs`

## Remaining Tasks

### 3. Optimize `var`/`std` with Welford's algorithm
**Current**: Two passes - compute mean, then compute variance
**Problem**: 2x memory bandwidth, 2x allocations for intermediates
**Fix**: Single-pass Welford's algorithm

```rust
// Welford's online algorithm:
// M1 = x1
// Mk = M(k-1) + (xk - M(k-1)) / k
// S1 = 0
// Sk = S(k-1) + (xk - M(k-1)) * (xk - Mk)
// variance = Sn / (n-1) or Sn / n
```

### 4. Fast path for scalar broadcast operations
**Current**: `x - x.mean()` allocates scalar array, broadcasts
**Fix**: Detect scalar in binary ops, use scalar directly

## Benchmark Results (After P0 fixes)

| Benchmark | Before | After | Status |
|-----------|--------|-------|--------|
| gradient (diff) | 0.03x | 1.00x | ✅ Fixed |
| second derivative | 0.03x | 1.16x | ✅ Fixed |
| diff + cumsum | 0.12x | 1.06x | ✅ Fixed |
| finite diff laplacian | 1.35x | 1.34x | ✅ Already good |
| cumsum | 1.18x | 1.04x | ✅ Already good |
| sigmoid | 0.91x | 0.91x | OK |
| variance axis | 0.18x | 0.18x | Needs Welford |
| matrix norm | 0.03x | 0.03x | Different issue |
