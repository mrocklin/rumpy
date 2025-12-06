# Temporary Array Elision

## Motivation

Chained operations like `x + 1 + 2 + 3` create intermediate arrays that are immediately discarded. Each allocation has overhead (memory bandwidth, cache pollution). NumPy optimizes this by detecting when an operand is "ephemeral" (refcount=1) and reusing its buffer for the result.

## Approach

Check Python refcount at the binding layer. If an operand has refcount=1 and meets reuse conditions, write the result into its buffer instead of allocating.

**Conditions for reuse:**
- Python refcount == 1
- Size >= 32768 elements (~256KB for f64)
- Same dtype as result (no type promotion)
- Same shape as result (after broadcast)
- C-contiguous with offset=0
- Arc::strong_count == 1 (no Rust views)

## Safety Tradeoff

NumPy uses expensive stack unwinding (~35μs) to verify no C extension holds a raw pointer to the buffer. We skip this and rely solely on refcount, accepting that:

1. Well-behaved C code increments refcounts when holding references
2. Most extensions use buffer protocol properly
3. Chained pure-Python ops (the target use case) don't involve C mid-chain

This is a pragmatic tradeoff for a new library where users are in pure-Python contexts.

## Size Threshold

Only apply to arrays >= 256KB. Small arrays are fast to allocate and the refcount check overhead isn't worth it. NumPy uses the same threshold because their stack unwinding cost requires it; we keep it for consistency and to avoid micro-optimization overhead.

## Implementation Notes

- Refcount checked via `pyo3::ffi::Py_REFCNT`
- `__add__(&self, ...)` provides immutable borrow; clone inner RumpyArray if Arc allows
- Prefer left operand for reuse, fallback to right if broadcasting allows
