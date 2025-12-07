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

**Registry dispatch**: Operations first check `UFuncRegistry` for optimized type-specific
loops. These loops receive stride information and handle contiguous/strided cases internally,
enabling LLVM auto-vectorization for contiguous arrays.

**Fallback**: If no registry loop exists, use `DTypeOps` trait methods (slower but universal).

**Broadcasting**: `map_binary_op` calls `broadcast_shapes()` then `broadcast_to()` on both
inputs before iteration. `broadcast_to()` creates a view with zero strides for
broadcast dimensions.

**Output**: Results are always contiguous (C-order), so we can write linearly.

## Adding New Operations

1. **Unary math** (sqrt, exp, log): Add variant to `UnaryOp` enum, implement in registry
2. **Binary ops**: Add variant to `BinaryOp` enum, register loops
3. **New reduction**: Add variant to `ReduceOp` enum, register strided loops
4. **Comparison ops**: Use `map_compare_op` with `ComparisonOp` enum

See `designs/adding-operations.md` for step-by-step guide.

## Performance

Registry loops enable SIMD via LLVM auto-vectorization:
- Contiguous fast path: convert to slice, tight loop
- Strided path: pointer arithmetic with stride offsets

See `designs/iteration-performance.md` for benchmarks.
