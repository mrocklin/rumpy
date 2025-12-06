# Iteration Performance

How rumpy iterates over array elements and the performance implications.

## Current Architecture

### Registry-Based Dispatch

All operations use the UFuncRegistry for type-specific loops:

```
Operation
    │
    ├─► Registry lookup ──► Typed loop function
    │                         └── Handles contiguous/strided internally
    │
    └─► Trait fallback (slower)
```

### Loop Types

| Type | Signature | Use Case |
|------|-----------|----------|
| `BinaryLoopFn` | `(a, b, out, n, strides)` | Element-wise binary ops |
| `UnaryLoopFn` | `(src, out, n, strides)` | Element-wise unary ops |
| `ReduceLoopFn` | `(acc, src, n, stride)` | Axis reductions |
| `ReduceAccFn` | `(acc, idx, val, offset)` | Full reductions (legacy) |

### Iteration Primitives

**BackstrideIter**: Efficient traversal using precomputed backstrides
- No per-element index→offset calculation
- Used for full-array operations

**AxisOffsetIter**: For axis operations
- Iterates over outer dimensions
- Yields base offsets for inner axis loop

## Performance Results

Benchmarks on 1000x1000 f64 array (vs NumPy):

| Operation | Contiguous Axis | Non-contiguous Axis |
|-----------|-----------------|---------------------|
| sum | **0.84x** (faster) | 3.5x slower |
| prod | **0.96x** (parity) | - |
| max | 1.3x slower | 3.6x slower |
| min | 1.3x slower | - |

Integer types often faster than NumPy (int32 sum: 0.3x = 3x faster).

## Key Optimizations

### 1. Multiple Accumulators

Breaking dependency chains enables SIMD:

```rust
// Bad: loop-carried dependency, no SIMD
for x in slice { acc += x; }

// Good: independent accumulators, SIMD-friendly
for chunk in slice.chunks_exact(8) {
    s0 += chunk[0]; s1 += chunk[1]; ...
}
acc = s0 + s1 + s2 + ...;
```

### 2. Contiguous Detection

Each loop checks `stride == itemsize` and uses slice-based iteration:

```rust
if stride == itemsize {
    let slice = from_raw_parts(ptr, n);
    // LLVM auto-vectorizes this
} else {
    // Strided pointer loop
}
```

### 3. Registry Dispatch

Type-specific loops avoid virtual dispatch overhead. Macros generate
optimized code for each dtype.

## Why Non-contiguous Is Slower

For axis=0 reduction on C-order array:
- Stride = 8000 bytes (jumping 1000 elements)
- Cache misses on every access
- No SIMD possible across non-contiguous memory

NumPy has similar issues but may use cache-aware blocking.

## Future Work

### Near-term

1. **Cache-aware blocking for non-contiguous**: Process in cache-friendly chunks
2. **Apply multi-accumulator to integers**: Currently only floats optimized
3. **Optimize max/min**: Current 1.3x gap may be due to NaN handling branches

### Medium-term

4. **Parallel reductions**: Rayon for large arrays
   - Split along outer dimension
   - Merge partial results

5. **Operation fusion**: Avoid intermediate allocations
   - `(a - mean).pow(2).sum()` in one pass
   - Requires Zip-style abstraction

### Long-term

6. **Explicit SIMD**: `std::simd` or `portable_simd` when stable
7. **Transpose for non-contiguous**: Copy to contiguous buffer when profitable
8. **BLAS integration**: Use optimized libraries for specific patterns

## Design Principles

1. **One pattern for all dtypes**: Macros generate typed loops, no special cases in operation code
2. **Contiguous fast path**: Every loop checks and optimizes for contiguous
3. **Registry consistency**: Reductions use same pattern as ufuncs
4. **Incremental optimization**: Add fast paths without breaking fallbacks
