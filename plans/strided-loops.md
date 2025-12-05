# Strided Inner Loops Plan

**Design doc:** `designs/strided-loops.md`
**Status:** Phase 2 complete

## Context

NumPy's inner loops receive stride info and iterate internally. This enables:
1. SIMD auto-vectorization for contiguous arrays (via slices)
2. Batched iteration for strided arrays (one call per inner row, not per element)

## Phase 1: Change Loop Signatures ✓

New signatures pass stride tuples:
```rust
type BinaryLoopFn = unsafe fn(*const u8, *const u8, *mut u8, usize, (isize, isize, isize));
type UnaryLoopFn = unsafe fn(*const u8, *mut u8, usize, (isize, isize));
```

## Phase 2: Strided Batching ✓

NumPy-style dispatch: iterate over all but innermost dimension, call loop once per row.

**Benchmark (1M elements, release build):**
| Case | NumPy | Rumpy | Ratio |
|------|-------|-------|-------|
| Contiguous Add | 0.76ms | 0.39ms | 0.51x (faster) |
| Strided 1D Add | 0.87ms | 0.76ms | 0.87x (faster) |

**Registration consolidation:** Nested macros reduced registry.rs from 1054 to 820 lines.

## Phase 3: Reductions

Reduce loops still use per-element signature. Future work:
- [ ] Update ReduceInitFn/ReduceAccFn to strided signatures
- [ ] Update reduce_all_op and reduce_axis_op dispatch

## Notes

- Contiguous: loop uses slices (LLVM auto-vectorizes)
- Strided: loop uses pointer arithmetic with actual strides
- Dispatch iterates outer dims, calls loop once per inner row
