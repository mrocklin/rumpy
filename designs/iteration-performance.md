# Iteration Performance

How rumpy iterates over array elements and the performance implications.

## Iteration Patterns

Rumpy has several ways to access array elements:

### 1. Registry Loops (Fast)

For ufuncs, strided loops receive pointers and strides directly:
```rust
unsafe fn add_f64(a: *const u8, b: *const u8, out: *mut u8, n: usize,
                  strides: (isize, isize, isize))
```

LLVM auto-vectorizes the contiguous case. See `designs/strided-loops.md`.

### 2. StridedIter (Medium)

Iterator yielding byte offsets for each element:
```rust
for offset in array.iter_offsets() {
    let val = ops.read_f64(ptr, offset);
}
```

Overhead: index array per iteration, branch per dimension to detect wraparound.
Used by: full-array reductions (sum, mean, var).

### 3. get_element (Slow)

Index-based element access:
```rust
for j in 0..axis_len {
    indices[axis] = j;
    let val = array.get_element(&indices);
}
```

Overhead: compute byte offset from indices each call.
Used by: axis reductions (sum_axis, var_axis, moment_axis).

## Performance Hierarchy

| Method | Relative Speed | Vectorizable |
|--------|---------------|--------------|
| Registry loop (contiguous) | 1.0x | Yes |
| Registry loop (strided) | 2-5x slower | No |
| StridedIter | 3-5x slower | No |
| get_element | 5-10x slower | No |

## Why Axis Reductions Are Slow

For a 1000x1000 array reduced along axis 0:
- We compute 1000 results
- Each result sums 1000 elements
- Each element access uses `get_element` (index→offset calculation)
- That's 1M index calculations vs NumPy's optimized strided iteration

## Optimization Strategy

### Contiguous Fast Paths

For contiguous f64, bypass iteration entirely:
```rust
if self.is_c_contiguous() && dtype.kind() == Float64 {
    let ptr = self.data_ptr() as *const f64;
    // Direct pointer arithmetic
}
```

### Strided Pointer Access

For reductions along an axis, compute the stride once and use pointer arithmetic:
```rust
let axis_stride = strides[axis];
let mut ptr = base_ptr;
for _ in 0..axis_len {
    let val = *ptr;
    ptr = ptr.offset(axis_stride);
}
```

This avoids per-element index calculation.

### Inner Loop Optimization

Process innermost dimension with a tight loop, only update outer indices:
```rust
for outer in outer_iter {
    let base = compute_base_offset(outer);
    for i in 0..inner_size {
        process(ptr.offset(base + i * inner_stride));
    }
}
```

## Current State

- Ufuncs: Use registry loops with strided support (fast)
- Full reductions: Use StridedIter (medium)
- Axis reductions: Use get_element (slow)

## Future Work

1. **Strided axis reductions**: Compute base pointer + axis stride, iterate with pointer arithmetic
2. **Parallel iteration**: Rayon for large arrays (requires thread-safe design)
3. **Explicit SIMD**: `std::simd` for critical paths when stable
