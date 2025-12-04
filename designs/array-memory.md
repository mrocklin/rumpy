<!-- AI: Read this before modifying array internals -->

# Array Memory Model

## Core Principle

RumpyArray uses `Arc<ArrayBuffer>` for shared ownership of memory. This enables
NumPy-style views without copying data.

## How Views Work

When creating a view (slice, reshape, transpose):
1. Clone the `Arc` (increment refcount, no data copy)
2. Compute new offset, shape, strides
3. Views share the same underlying buffer

```
Original: buffer=Arc(data), offset=0, shape=[10], strides=[8]
View[2:7]: buffer=Arc(data), offset=16, shape=[5], strides=[8]  <- same Arc
```

## When Copies Are Needed

- Type conversion (float64 → float32)
- Negative step on non-contiguous array
- Advanced indexing (arr[[1,3,5]])
- Explicit copy request

## Thread Safety

`Arc` provides Send + Sync. Multiple reads are safe. For mutation:
- Use `Arc::get_mut()` to get exclusive access
- Or call `make_owned()` to copy if shared

## Strides

Strides are `isize` (signed) to support negative strides for reversed views.
C-order strides: last dimension = itemsize, decreasing toward first dimension.
