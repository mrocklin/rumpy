# Hybrid UFunc Dispatch Plan

**Design doc:** `designs/dtype-system.md`
**Status:** Planning - ready for Phase 1

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
- [ ] Define `TypeSignature` struct
- [ ] Define `InnerLoopFn` type (or trait)
- [ ] Create `UFuncRegistry` with register/lookup
- [ ] Add global or lazy-static registry instance
- [ ] Write tests for registry lookup

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
- [ ] Modify `map_binary_op` to check registry
- [ ] Register same-type loops for f64, f32, i64, i32, etc.
- [ ] Verify all existing tests pass
- [ ] Add test for registry-dispatched operation

## Phase 3: Unary Ops (future)

Same pattern for `map_unary_op`.

## Phase 4: Reductions (future)

Same pattern for `reduce_all_op`, `reduce_axis_op`.

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

## Open Decisions

1. **Global registry** (lazy_static) vs **passed registry** - leaning global for simplicity
2. **InnerLoopFn signature** - needs refinement for strided access
3. **Promotion** - keep in `map_binary_op` or move to registry lookup
