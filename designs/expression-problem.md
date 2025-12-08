# Expression Problem Architecture

How to handle N dtypes × M operations without O(N×M) code explosion, while maintaining performance.

## The Four Dimensions

Every array operation has four orthogonal concerns:

```
1. DType:    float64, int32, complex128, ...
2. Operation: add, mul, sum, sqrt, ...
3. Layout:    contiguous, strided, scalar-broadcast
4. SIMD:      scalar, SSE, AVX2, AVX512, NEON
```

Good architecture addresses each independently. Bad architecture conflates them.

## NumPy's Approach

NumPy separates concerns through three tiers:

1. **Type resolution**: Determine input/output dtypes, handle promotion
2. **Loop selection**: Find best implementation via ArrayMethod dispatch
3. **Loop execution**: Run the computation (SIMD-specialized per CPU target)

Build-time code generation (`.c.src` templates) creates dtype-specific loops.
Multi-target compilation creates architecture-specific SIMD variants.
Runtime dispatch selects the best combination.

## Current Rumpy Approach

Rumpy uses two-tier dispatch:
- **Registry**: HashMap of `(Op, TypeSignature) → LoopFn` for fast paths
- **Trait fallback**: `DTypeOps` trait methods for universal coverage

Each registered loop embeds all concerns:
```rust
|a_ptr, b_ptr, out_ptr, n, strides| {
    if contiguous { /* SIMD-friendly path */ }
    else { /* strided path */ }
}
```

This works but leads to repetition: every loop re-implements contiguous detection and SIMD patterns.

## Recommended Principles

### Factor Out Layout

Separate "what operation" from "how to traverse memory":

```rust
// Operation knows the math
fn add(a: f64, b: f64) -> f64 { a + b }

// Layout strategy knows memory traversal
fn map_contiguous<F>(src: &[T], dst: &mut [T], f: F);
fn map_strided<F>(src: *const u8, dst: *mut u8, n: usize, stride: isize, f: F);
```

SIMD optimizations live in `map_contiguous`, applied once for all operations.

### SIMD as Layout Property

Contiguous arrays enable SIMD. This is a property of memory layout, not operations.

```rust
fn map_contiguous<T, F>(data: &[T], f: F) {
    // Single implementation handles SIMD for ALL operations
    #[cfg(target_feature = "avx2")]
    if TypeId::of::<T>() == TypeId::of::<f64>() && size >= 32 {
        return simd_map_f64(data, f);
    }
    data.iter().map(f)
}
```

### Preserve Two-Tier Dispatch

The registry + trait fallback pattern is sound:
- Registry provides performance for hot paths
- Trait fallback ensures all dtype combinations work
- Adding a new dtype automatically works (via trait impl)

### Consider Trait-Based Operations

Instead of `enum UnaryOp { Sqrt, Exp, ... }`:

```rust
trait UnaryFn<T>: Copy { fn apply(self, v: T) -> T; }
struct Sqrt;
impl UnaryFn<f64> for Sqrt { fn apply(self, v: f64) -> f64 { v.sqrt() } }
```

This enables static dispatch and inlining while maintaining extensibility.

## Performance Implications

Orthogonal design often improves performance because:

1. **Shared optimization**: SIMD code written once, used by all ops
2. **Better inlining**: Monomorphization through traits vs dynamic dispatch
3. **Clearer hot paths**: Specialized contiguous loops without branching

The current 1.3x gap on max/min likely comes from branching in every iteration.
Factored SIMD would handle this uniformly.

## See Also

- `designs/ufuncs.md` - ufunc dispatch architecture
- `designs/iteration-performance.md` - current performance analysis
- NumPy: `numpy/_core/src/umath/dispatching.cpp`
