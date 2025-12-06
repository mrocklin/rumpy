# Backstride Iteration

Core iteration primitives for efficient strided array traversal.

## Problem

Array operations need to iterate over elements efficiently. Key challenges:
1. **Strided access**: Non-contiguous memory patterns
2. **Axis reductions**: Sum/prod/max along one dimension
3. **Type genericity**: Same pattern for f64, f32, i64, etc.

## Design Principles

1. **One pattern for all dtypes**: No special-case code per type
2. **Contiguous fast path**: SIMD when stride == itemsize
3. **Registry-based dispatch**: Consistent with ufunc architecture

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

### ReduceLoopFn

Registry function type for strided reductions:

```rust
type ReduceLoopFn = unsafe fn(
    acc_ptr: *mut u8,
    src_ptr: *const u8,
    n: usize,
    stride: isize,
);
```

Each loop handles contiguous/strided internally:
```rust
if stride == itemsize {
    let slice = from_raw_parts(src_ptr, n);
    *acc = slice.iter().sum();  // SIMD
} else {
    // Strided pointer loop
}
```

## Usage

```rust
// reduce_axis_op uses registry strided loops
for (i, base_offset) in arr.axis_offsets(axis).enumerate() {
    let src = src_ptr.offset(base_offset);
    let acc = result_ptr.add(i * itemsize);
    loop_fn(acc, src, axis_len, axis_stride);
}
```

## Architecture

```
reduce_axis_op
    │
    ├─► lookup_reduce_strided() ──► ReduceLoopFn (fast)
    │                                  └── contiguous/strided handled internally
    │
    ├─► lookup_reduce() ──► ReduceAccFn (per-element, slower)
    │
    └─► trait dispatch (fallback)
```

## Non-Goals

- **Zip/fusion**: Defer until needed
- **Parallelism**: Rayon can layer on top later
- **Explicit SIMD**: Rely on LLVM auto-vectorization
