# Strided Inner Loops

Memory traversal patterns for contiguous and strided arrays.

## Core Insight

Operations need both contiguous (SIMD-friendly) and strided (view-friendly) paths.
The kernel/dispatch system separates these in `loops/`:

```
loops/
├── contiguous.rs  # Slice-based, LLVM auto-vectorizes
└── strided.rs     # Pointer arithmetic with byte offsets
```

## Loop Functions

```rust
// Contiguous: operates on slices
pub fn map_binary<T: Copy, K: BinaryKernel<T>>(
    a: &[T], b: &[T], out: &mut [T], _kernel: K
) {
    for i in 0..a.len() {
        out[i] = K::apply(a[i], b[i]);
    }
}

// Strided: pointer arithmetic
pub unsafe fn map_binary_strided<T: Copy, K: BinaryKernel<T>>(
    a_ptr: *const T, a_stride: isize,
    b_ptr: *const T, b_stride: isize,
    out_ptr: *mut T, out_stride: isize,
    n: usize, _kernel: K
) {
    for i in 0..n {
        let offset = i as isize;
        let a = *a_ptr.byte_offset(a_stride * offset);
        let b = *b_ptr.byte_offset(b_stride * offset);
        *out_ptr.byte_offset(out_stride * offset) = K::apply(a, b);
    }
}
```

## Dispatch Selects Layout

Layout detection happens once in `dispatch.rs`:

```rust
if a.is_contiguous() && b.is_contiguous() && out.is_contiguous() {
    loops::map_binary(a_slice, b_slice, out_slice, kernel);
} else {
    unsafe { loops::map_binary_strided(a_ptr, a_stride, ..., kernel); }
}
```

## Benefits

1. **SIMD for contiguous**: Slice loops auto-vectorize
2. **Correct for strided**: Pointer arithmetic handles views
3. **Single kernel definition**: Same `K::apply` used by both paths
4. **Monomorphization**: Zero-sized kernels inline completely

## See Also

- `designs/kernel-dispatch.md` - Full architecture
- `designs/iteration-performance.md` - Benchmarks
