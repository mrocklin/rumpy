# Iteration Performance

How rumpy iterates over array elements and the performance implications.

## Current Architecture

### Kernel/Dispatch System

All operations use the kernel/dispatch architecture:

```
Operation
    │
    ├─► dispatch.rs (type resolution, layout detection)
    │       │
    │       ├─► Contiguous: loops/contiguous.rs (slice-based, SIMD-friendly)
    │       │
    │       └─► Strided: loops/strided.rs (pointer arithmetic)
    │
    └─► Trait fallback (slower, universal)
```

See `designs/kernel-dispatch.md` for architecture details.

### Loop Functions

| Location | Pattern | Use Case |
|----------|---------|----------|
| `loops/contiguous.rs` | Slice iteration | Contiguous arrays (SIMD) |
| `loops/strided.rs` | Pointer arithmetic | Non-contiguous views |

### Iteration Primitives

**BackstrideIter**: Efficient traversal using precomputed backstrides
- No per-element index→offset calculation
- Used for full-array operations

**AxisOffsetIter**: For axis operations
- Iterates over outer dimensions
- Yields base offsets for inner axis loop

## Performance Results

Benchmarks on 1000x1000 f64 array (vs NumPy):

| Operation | Contiguous | Non-contiguous Axis |
|-----------|------------|---------------------|
| sum | **0.84x** (faster) | 3.5x slower |
| prod | **0.96x** (parity) | - |
| max | 1.3x slower | 3.6x slower |
| min | 1.3x slower | - |

Integer types often faster than NumPy (int32 sum: 0.3x = 3x faster).

## Key Optimizations

### 1. Multiple Accumulators (8-way)

Breaking dependency chains enables SIMD. Used in `loops/contiguous.rs`:

```rust
// Bad: loop-carried dependency, no SIMD
for x in slice { acc += x; }

// Good: 8 independent accumulators
let mut r = [K::combine(K::init(), data[0]), ...]; // 8 accumulators
for chunk in data.chunks_exact(8) {
    r[0] = K::combine(r[0], chunk[0]);
    r[1] = K::combine(r[1], chunk[1]);
    // ...
}
// Tree reduction at end
```

### 2. Layout Detection in Dispatch

Layout selection happens once in `dispatch.rs`, not per-element:

```rust
if contiguous {
    loops::map_binary(a_slice, b_slice, out_slice, kernel);
} else {
    unsafe { loops::map_binary_strided(..., kernel); }
}
```

### 3. Zero-Sized Kernel Types

Kernels are structs with no fields. `K::apply(a, b)` monomorphizes to inline code:

```rust
pub struct Add;
impl BinaryKernel<f64> for Add {
    fn apply(a: f64, b: f64) -> f64 { a + b }  // Inlined
}
```

## Why Non-contiguous Is Slower

For axis=0 reduction on C-order array:
- Stride = 8000 bytes (jumping 1000 elements)
- Cache misses on every access
- No SIMD possible across non-contiguous memory

NumPy has similar issues but may use cache-aware blocking.

## Future Work

### Near-term

1. **Cache-aware blocking for non-contiguous**: Process in cache-friendly chunks
2. **Explicit SIMD for reductions**: Current 1.3x gap on max/min

### Medium-term

3. **Parallel reductions**: Rayon for large arrays
   - Split along outer dimension
   - Merge partial results

4. **Operation fusion**: Avoid intermediate allocations
   - `(a - mean).pow(2).sum()` in one pass
   - Requires Zip-style abstraction

### Long-term

5. **Explicit SIMD**: `std::simd` or `portable_simd` when stable
6. **Transpose for non-contiguous**: Copy to contiguous buffer when profitable
7. **BLAS integration**: Use optimized libraries for specific patterns

## Design Principles

1. **Orthogonal concerns**: Operations, layouts, dtypes factored separately
2. **Contiguous fast path**: Layout detection in dispatch, not loops
3. **Monomorphization**: Zero-sized kernels enable inlining
4. **Incremental optimization**: Add fast paths without breaking fallbacks
