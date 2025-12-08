<!-- AI: Read this before adding new operations -->

# Ufunc Design

## Core Principle

All element-wise operations and reductions use shared iteration machinery.
This keeps code DRY and ensures consistent handling of strides, broadcasting, and dtypes.

## The Four Primitives

```
map_unary_op(arr, UnaryOp)       -> Result<RumpyArray>  # sqrt, exp, log, etc.
map_binary_op(a, b, BinaryOp)   -> Result<RumpyArray>  # +, -, *, / with broadcasting
reduce_all_op(arr, ReduceOp)    -> f64                 # fold all elements
reduce_axis_op(arr, axis, ReduceOp) -> RumpyArray      # fold along one axis
```

## How They Work

**Kernel/dispatch architecture**: Operations use the orthogonal kernel/loop/dispatch system.
See `designs/kernel-dispatch.md` for details.

```
ufunc.rs (public API)
    → dispatch.rs (type resolution, layout detection)
        → kernels/*.rs (operation definitions: Add, Sum, Sqrt, etc.)
        → loops/*.rs (memory traversal: contiguous SIMD, strided pointer)
```

**Fallback**: If dispatch doesn't handle a dtype, falls back to `DTypeOps` trait methods.

**Broadcasting**: `map_binary_op` calls `broadcast_shapes()` then `broadcast_to()` on both
inputs before iteration. `broadcast_to()` creates a view with zero strides for
broadcast dimensions.

**Output**: Results are always contiguous (C-order), so we can write linearly.

## Adding New Operations

See `designs/kernel-dispatch.md` for the recommended approach:
1. Add kernel in `kernels/*.rs`
2. Add dispatch in `dispatch.rs`
3. Wire up in `ufunc.rs`

## Performance

Kernels are zero-sized types that monomorphize to tight code.
Layout selection happens once in dispatch, then runs specialized loops:
- **Contiguous**: Slice-based, LLVM auto-vectorizes
- **Strided**: Pointer arithmetic with byte offsets

See `designs/iteration-performance.md` for benchmarks.
