<!-- AI: Read this before adding new operations -->

# Ufunc Design

## Core Principle

All element-wise operations and reductions use shared iteration machinery.
This keeps code DRY and ensures consistent handling of strides, broadcasting, and dtypes.

## The Four Primitives

```
map_unary(arr, f)         -> RumpyArray   # f(x) for each element
map_binary(a, b, f)       -> RumpyArray   # f(a, b) with broadcasting
reduce_all(arr, init, f)  -> f64          # fold all elements
reduce_axis(arr, axis, init, f) -> RumpyArray  # fold along one axis
```

## How They Work

All use index-based iteration via `get_element(&indices)` + `increment_indices()`.
This correctly handles non-contiguous arrays (views, transposes, broadcasts).

**Broadcasting**: `map_binary` calls `broadcast_shapes()` then `broadcast_to()` on both
inputs before iteration. `broadcast_to()` creates a view with zero strides for
broadcast dimensions.

**Output**: Results are always contiguous (C-order), so we can write linearly.

## Adding New Operations

1. **Unary math** (sqrt, exp, log): Add method using `map_unary`
2. **Binary ops**: Already handled via `BinaryOp` enum
3. **New reduction**: Add method using `reduce_all` or `reduce_axis`
4. **Comparison ops**: Would use `map_binary` with bool output dtype

## Performance Notes

Current implementation is simple but not optimal:
- Index-based iteration has overhead vs pointer arithmetic
- No SIMD vectorization
- No special fast path for contiguous arrays

These are intentional tradeoffs for correctness and simplicity. Optimize later
by adding contiguous fast paths that fall back to index-based for views.
