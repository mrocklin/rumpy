# Hybrid UFunc Dispatch Plan

**Design doc:** `designs/dtype-system.md`
**Status:** Phase 1-4 complete

## Context

We currently use a trait-based approach where each dtype implements `DTypeOps` with methods like `binary_op`, `unary_op`, etc. This works but has limitations:
- Operations that don't make sense for all dtypes (bitwise on floats)
- Return types that vary by input (complex abs → real)
- No way to register optimized loops for specific type combinations

NumPy uses table-based dispatch where each ufunc has registered "inner loops" for specific type signatures. We want a hybrid: registry-first with trait fallback.

## Current Architecture

Key files:
- `src/array/dtype/mod.rs` - `DTypeOps` trait, `UnaryOp`/`BinaryOp`/`ReduceOp` enums
- `src/ops/mod.rs` - `map_unary_op`, `map_binary_op`, reduction functions
- `src/array/dtype/*.rs` - Per-dtype implementations

Current dispatch flow (in `map_binary_op`):
```rust
let same_dtype = a.dtype() == b.dtype() && a.dtype() == result.dtype();
if same_dtype {
    // Direct buffer op via trait
    result_ops.binary_op(op, a_ptr, a_offset, b_ptr, b_offset, result_ptr, i);
} else {
    // Mixed types: read as f64, operate, write back
    let av = a_ops.read_f64(...);
    let bv = b_ops.read_f64(...);
    // ... compute ...
    result_ops.write_f64(...);
}
```

## Target Architecture

```
UFuncRegistry
├── loops: HashMap<(Op, TypeSignature), InnerLoopFn>
└── fallback: current DTypeOps-based dispatch

Dispatch order:
1. Exact match in registry → use registered loop
2. No match → fall back to dtype.ops().binary_op(...)
```

## Phase 1: Infrastructure

Create the registry system without changing existing behavior.

Files to create/modify:
- `src/ops/registry.rs` (new) - UFuncRegistry, TypeSignature, InnerLoopFn

```rust
// Sketch of API
pub struct TypeSignature {
    inputs: Vec<DTypeKind>,
    output: DTypeKind,
}

pub type InnerLoopFn = fn(
    inputs: &[(*const u8, isize)],  // (ptr, byte_offset) pairs
    output: (*mut u8, usize),        // (ptr, element_index)
);

pub struct UFuncRegistry {
    loops: HashMap<(BinaryOp, TypeSignature), InnerLoopFn>,
}

impl UFuncRegistry {
    pub fn register_binary(&mut self, op: BinaryOp, sig: TypeSignature, f: InnerLoopFn);
    pub fn lookup_binary(&self, op: BinaryOp, a: &DType, b: &DType) -> Option<(InnerLoopFn, DType)>;
}
```

Tasks:
- [x] Define `TypeSignature` struct
- [x] Define `InnerLoopFn` type (or trait)
- [x] Create `UFuncRegistry` with register/lookup
- [x] Add global registry instance (using `OnceLock<RwLock<...>>`)
- [x] Write tests for registry lookup

## Phase 2: Wire Up Binary Ops

Modify `map_binary_op` to check registry first.

```rust
fn map_binary_op(a: &RumpyArray, b: &RumpyArray, op: DTypeBinaryOp) -> Option<RumpyArray> {
    // NEW: Check registry first
    if let Some((loop_fn, out_dtype)) = REGISTRY.lookup_binary(op, &a.dtype(), &b.dtype()) {
        return Some(apply_binary_loop(a, b, out_dtype, loop_fn));
    }

    // Existing fallback code...
}
```

Tasks:
- [x] Modify `map_binary_op` to check registry
- [x] Register same-type loops for f64, f32, i64, i32
- [x] Verify all 325 existing tests pass
- [x] Registry tests verify dispatch (6 tests in registry.rs)

## Phase 3: Unary Ops

Same pattern for `map_unary_op`.

Tasks:
- [x] Modify `map_unary_op` to check registry first
- [x] Register unary loops: Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Tan for f64
- [x] Register Neg, Abs, Sqrt for f32; Neg, Abs for i64, i32
- [x] All 352 tests pass (27 new dtype interaction tests)

## Phase 4: Reductions

Same pattern for `reduce_all_op`, `reduce_axis_op`.

Tasks:
- [x] Add `ReduceInitFn` and `ReduceAccFn` types to registry
- [x] Wire up `reduce_all_op` and `reduce_axis_op` to check registry
- [x] Register reduction loops for f64, f32, i64, i32 (Sum, Prod, Max, Min)
- [x] All 352 tests pass

## More Dtype Loops

Registered loops for all remaining dtypes:
- [x] uint8, uint32, uint64 binary ops (Add, Sub, Mul, Div)
- [x] bool binary ops (Add=or, Mul=and)
- [x] complex128 binary ops (Add, Sub, Mul, Div)
- [x] uint unary ops (Abs only - no Neg for unsigned)
- [x] complex128 unary ops (Neg)
- [x] uint reduce ops (Sum, Prod, Max, Min)
- [x] bool reduce ops (Sum=any, Prod=all)
- [x] complex128 reduce ops (Sum, Prod - no Max/Min)

## Datetime Validation

- [x] datetime + datetime returns TypeError
- [x] datetime * anything returns TypeError
- [x] datetime / anything returns TypeError
- [x] Proper error types: ShapeMismatch -> ValueError, UnsupportedDtype -> TypeError
- [x] All 353 tests pass (former xfail test now passes)

## What's Next

Potential next steps (in rough priority order):
1. **New operations** - Bitwise ops, floor/ceil, more trig (asin, acos, atan)
2. **Performance** - Consider SIMD loops for hot paths

## Why Hybrid?

| Scenario | Pure Trait | Pure Registry | Hybrid |
|----------|------------|---------------|--------|
| Add new dtype | ✓ Just impl trait | ✗ Register everywhere | ✓ Fallback works |
| Add new op | ✗ Change trait | ✓ Just register | ✓ Can do either |
| Optimize hot path | ✗ All or nothing | ✓ Register fast loop | ✓ Register fast loop |
| Type-specific behavior | ✗ Awkward | ✓ Natural | ✓ Natural |

## Testing Strategy

1. All 325 existing tests must pass (fallback works)
2. New tests verify registry dispatch is used when available
3. Test mixed-type operations still work via fallback

## Decisions Made

1. **Global registry** - Using `OnceLock<RwLock<UFuncRegistry>>` (std, no external deps)
2. **InnerLoopFn signature** - `fn(a_ptr, a_offset, b_ptr, b_offset, out_ptr, out_idx)` - works with strided arrays
3. **Promotion** - NumPy-compatible rules in `promote_dtype()`:
   - int32/int64 + float32 -> float64 (safe casting)
   - int8/int16 + float32 -> float32
   - complex + anything -> complex128
   - Mixed-type ops use complex path if result is complex

## Additional Work Done

- **Type promotion tests** (`tests/test_dtype_interactions.py`): 27 tests verifying NumPy compatibility
- **Complex mixed-type ops**: Fixed to use `read_complex`/`write_complex` when result is complex
