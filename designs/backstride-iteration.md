# Backstride Iteration

Core iteration primitives for efficient strided array traversal.

## Problem

Array operations need to iterate over elements efficiently. Key challenges:
1. **Strided access**: Non-contiguous memory patterns
2. **Axis reductions**: Sum/prod/max along one dimension
3. **Type genericity**: Same pattern for f64, f32, i64, etc.

## Design Principles

1. **Orthogonal concerns**: Operations, layouts, dtypes factored separately
2. **Contiguous fast path**: Layout detection in dispatch, not per-element
3. **Kernel monomorphization**: Zero-sized types enable inlining

## Components

### BackstrideIter

Efficient traversal using precomputed backstrides:

```rust
struct BackstrideIter {
    offset: isize,
    counters: Vec<usize>,
    strides: Vec<isize>,
    backstrides: Vec<isize>,  // (shape[i]-1) * stride[i]
}
```

No per-element index→offset calculation.

### AxisOffsetIter

For axis operations, iterate over outer dimensions:

```rust
arr.axis_offsets(axis) -> impl Iterator<Item = isize>
```

Each offset is a base pointer for reducing along the axis.

## Dispatch Architecture

```
reduce_axis_op
    │
    ├─► dispatch::dispatch_reduce_axis_*() ──► Kernel + Loop
    │       │
    │       ├─► Contiguous axis: loops::reduce() on slice
    │       │
    │       └─► Strided: loops::reduce_strided() with pointers
    │
    └─► trait fallback (Bool, DateTime64)
```

## Usage

```rust
// dispatch handles layout selection
dispatch_reduce_axis_sum(arr, axis, out_shape)
    → match dtype {
        Float64 => dispatch_reduce_axis_typed::<f64, Sum>(...)
        // ...
    }
```

Inside `dispatch_reduce_axis_typed`:
```rust
if axis_stride == itemsize {
    // Contiguous: use slice-based reduce
    loops::reduce(slice, kernel)
} else {
    // Strided: pointer arithmetic
    unsafe { loops::reduce_strided(ptr, stride, n, kernel) }
}
```

## Non-Goals

- **Zip/fusion**: Defer until needed
- **Parallelism**: Rayon can layer on top later
- **Explicit SIMD**: Rely on LLVM auto-vectorization for now

## See Also

- `designs/kernel-dispatch.md` - Full architecture
- `designs/strided-loops.md` - Loop implementation details
