# Strided Inner Loops Plan

**Design doc:** `designs/strided-loops.md`
**Status:** Phase 1 complete

## Context

Current loop functions operate on single elements with byte offsets. The dispatch
code computes offsets per-element, preventing SIMD. NumPy's approach passes strides
to the loop, letting it iterate internally with potential fast paths.

## Phase 1: Change Loop Signatures ✓

Modify registry to use strided loop signatures.

Files modified:
- `src/ops/registry.rs` - new function types, macros with contiguous fast path
- `src/ops/mod.rs` - update dispatch to pass strides

New signatures:
```rust
type BinaryLoopFn = unsafe fn(*const u8, *const u8, *mut u8, usize, (isize, isize, isize));
type UnaryLoopFn = unsafe fn(*const u8, *mut u8, usize, (isize, isize));
```

Tasks:
- [x] Update BinaryLoopFn signature in registry.rs
- [x] Update UnaryLoopFn signature in registry.rs
- [x] Create macros for strided loops with contiguous fast path
- [x] Update map_binary_op to pass strides and call loop once
- [x] Update map_unary_op to pass strides and call loop once
- [x] Update all registered loops to new signature
- [x] Update registry tests
- [x] All 372 tests pass

## Phase 2: Contiguous Fast Path ✓

Internal contiguous detection was added as part of Phase 1. The macros
`register_strided_binary!` and `register_strided_unary!` include a contiguous
fast path that uses slices (LLVM auto-vectorizes these).

Tasks:
- [x] Create macro for binary loops with contiguous fast path
- [x] Create macro for unary loops with contiguous fast path
- [x] Re-register loops using macros
- [ ] Benchmark contiguous vs strided performance

## Phase 3: Reductions

Same pattern for reduce loops. Not yet implemented - reduce loops still use
the per-element signature.

Tasks:
- [ ] Update ReduceInitFn and ReduceAccFn signatures
- [ ] Update reduce_all_op and reduce_axis_op
- [ ] Re-register reduction loops

## Notes

- Keep trait-based fallback for dtypes without registered loops
- Strided case uses pointer arithmetic
- Contiguous case uses slice iteration (LLVM auto-vectorizes)
- Dispatch checks `is_c_contiguous()` to choose fast path vs per-element path
