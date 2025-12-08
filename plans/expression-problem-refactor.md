# Plan: Expression Problem Refactor

Design: `designs/expression-problem.md`

## Goal

Refactor the ufunc system to achieve orthogonal separation of:
1. **Operations** - the math (add, sqrt, sum)
2. **Layouts** - memory traversal (contiguous, strided)
3. **DTypes** - element types (f64, i32)

Original `registry.rs` was 1840 lines of repetitive macros embedding all three concerns.

## Current Architecture (IMPLEMENTED)

```
src/ops/
├── kernels/           # Pure operation definitions
│   ├── mod.rs         # Traits: UnaryKernel, BinaryKernel, ReduceKernel, CompareKernel
│   ├── arithmetic.rs  # Add, Sub, Mul, Div, Sum, Prod, Max, Min per dtype
│   ├── comparison.rs  # Gt, Lt, Ge, Le, Eq, Ne per dtype
│   └── math.rs        # Sqrt, Exp, Sin, etc per dtype
│
├── loops/             # Layout strategies (SIMD lives here)
│   ├── mod.rs         # Re-exports
│   ├── contiguous.rs  # SIMD-friendly traversal (map_binary, map_unary, map_compare, reduce)
│   └── strided.rs     # Pointer arithmetic traversal
│
├── dispatch.rs        # Type resolution + layout detection
├── registry.rs        # Legacy loops (axis reductions, bitwise)
└── ufunc.rs           # Orchestration (tries dispatch first, falls back to registry/trait)
```

Benefits achieved:
- Adding operation: 1 file (kernels/*.rs)
- Adding dtype: impl kernel traits
- SIMD optimization: change loops/contiguous.rs once

## Target Architecture

```
src/ops/
├── kernels/           # Pure operation definitions
│   ├── mod.rs         # Traits: UnaryKernel, BinaryKernel, ReduceKernel
│   ├── arithmetic.rs  # Add, Sub, Mul, Div per dtype
│   ├── math.rs        # Sqrt, Exp, Sin, etc per dtype
│   └── reductions.rs  # Sum, Prod, Max, Min per dtype
│
├── loops/             # Layout strategies (SIMD lives here)
│   ├── mod.rs         # Traits: ContiguousLoop, StridedLoop
│   ├── contiguous.rs  # SIMD-optimized traversal
│   └── strided.rs     # Pointer arithmetic traversal
│
├── dispatch.rs        # Registry + type resolution (slimmed)
└── ufunc.rs           # map_unary_op, map_binary_op (orchestration)
```

## Phase 1: Kernel Traits

Create operation traits that are separate from dtype:

```rust
// src/ops/kernels/mod.rs
pub trait BinaryKernel<T>: Copy {
    fn apply(a: T, b: T) -> T;
    const IDENTITY: Option<T> = None;  // For reductions
}

pub trait UnaryKernel<T>: Copy {
    fn apply(v: T) -> T;
}

pub trait ReduceKernel<T>: Copy {
    fn init() -> T;
    fn combine(acc: T, v: T) -> T;
}
```

```rust
// src/ops/kernels/arithmetic.rs
pub struct Add;
impl BinaryKernel<f64> for Add { fn apply(a: f64, b: f64) -> f64 { a + b } }
impl BinaryKernel<f32> for Add { fn apply(a: f32, b: f32) -> f32 { a + b } }
impl BinaryKernel<i64> for Add { fn apply(a: i64, b: i64) -> i64 { a.wrapping_add(b) } }
// ... macro can generate these
```

**Why traits not closures**: Traits enable monomorphization. The loop `for i in 0..n { out[i] = K::apply(a[i], b[i]) }` compiles to tight code per (kernel, dtype) pair.

## Phase 2: Loop Strategies

Factor out memory traversal with SIMD in one place:

```rust
// src/ops/loops/contiguous.rs
pub fn map_binary<T, K: BinaryKernel<T>>(
    a: &[T], b: &[T], out: &mut [T], _kernel: K
) {
    // Single SIMD implementation for ALL binary ops
    #[cfg(target_feature = "avx2")]
    if size_of::<T>() == 8 && a.len() >= 32 {
        return simd_map_binary::<T, K>(a, b, out);
    }

    // Scalar fallback - still vectorizes via LLVM for simple ops
    for i in 0..a.len() {
        out[i] = K::apply(a[i], b[i]);
    }
}

pub fn reduce<T, K: ReduceKernel<T>>(data: &[T], _kernel: K) -> T {
    // 8-accumulator pattern, written once
    let (mut s0, mut s1, ...) = (K::init(), K::init(), ...);
    for chunk in data.chunks_exact(8) {
        s0 = K::combine(s0, chunk[0]);
        // ...
    }
    K::combine(K::combine(s0, s1), K::combine(s2, s3))
}
```

```rust
// src/ops/loops/strided.rs
pub unsafe fn map_binary_strided<T, K: BinaryKernel<T>>(
    a_ptr: *const T, a_stride: isize,
    b_ptr: *const T, b_stride: isize,
    out_ptr: *mut T, out_stride: isize,
    n: usize, _kernel: K
) {
    for i in 0..n {
        let a = *a_ptr.byte_offset(a_stride * i as isize);
        let b = *b_ptr.byte_offset(b_stride * i as isize);
        *out_ptr.byte_offset(out_stride * i as isize) = K::apply(a, b);
    }
}
```

## Phase 3: Slim Dispatch

Registry becomes a lookup table, not implementation storage:

```rust
// src/ops/dispatch.rs
pub fn dispatch_binary<K: BinaryKernel<f64> + BinaryKernel<f32> + ...>(
    a: &RumpyArray, b: &RumpyArray, out: &mut RumpyArray, kernel: K
) {
    match a.dtype().kind() {
        DTypeKind::Float64 if a.is_contiguous() && b.is_contiguous() => {
            loops::contiguous::map_binary(a.as_slice(), b.as_slice(), out.as_mut_slice(), kernel);
        }
        DTypeKind::Float64 => {
            unsafe { loops::strided::map_binary_strided(..., kernel); }
        }
        // Other dtypes...
    }
}
```

**Key change**: Layout selection happens ONCE in dispatch, not embedded in every kernel.

## Phase 4: Backward Compat

Keep `BinaryOp` enum as public API, implement via dispatch to kernels:

```rust
// src/ops/ufunc.rs
pub fn map_binary_op(a: &RumpyArray, b: &RumpyArray, op: BinaryOp) -> Result<RumpyArray> {
    match op {
        BinaryOp::Add => dispatch_binary(a, b, &mut out, kernels::Add),
        BinaryOp::Sub => dispatch_binary(a, b, &mut out, kernels::Sub),
        // ...
    }
}
```

`DTypeOps` trait can be slimmed - operation implementations move to kernels.

## Execution Order

### Stream 1: Foundation (no breaking changes) - COMPLETE
1. ✅ Create `src/ops/kernels/mod.rs` with traits (BinaryKernel, UnaryKernel, ReduceKernel)
2. ✅ Create `src/ops/kernels/arithmetic.rs` with Add/Sub/Mul/Div/Sum/Prod
3. ✅ Create `src/ops/loops/mod.rs` with loop traits
4. ✅ Create `src/ops/loops/contiguous.rs` with generic loops (4-accumulator reduce)

### Stream 2: Migration - PARTIAL
5. ✅ Add `dispatch_binary` that uses new kernels for f64/f32 (and all int types)
6. ✅ Update `map_binary_op` to use dispatch for Add/Sub/Mul/Div
   - Performance verified: ~0.99x NumPy on 1M element arrays
7. ✅ Migrate unary ops (Neg, Abs, Square, Sqrt, Exp, Log, Sin, Cos, Tan, Floor, Ceil)
8. ✅ Migrate reduce ops (Sum, Prod, Max, Min)

### Stream 3: Complete Migration - MOSTLY DONE

#### 3a. ✅ All Binary Ops migrated (18 ops)
- Add, Sub, Mul, Div, Pow, Mod, FloorDiv, Maximum, Minimum
- Arctan2, Hypot, FMax, FMin, Copysign, Logaddexp, Logaddexp2, Nextafter

#### 3b. ✅ All Unary Ops migrated (31 ops)
- Neg, Abs, Square, Sqrt, Exp, Log, Log10, Log2, Sin, Cos, Tan, Floor, Ceil
- Sinh, Cosh, Tanh, Arcsin, Arccos, Arctan, Sign, Positive, Reciprocal
- Exp2, Expm1, Log1p, Cbrt, Trunc, Rint, Arcsinh, Arccosh, Arctanh
- (Isnan, Isinf, Isfinite, Signbit return bool - handled by trait fallback)

#### 3c. ✅ All Reduce Ops migrated (4 ops)
- Sum, Prod, Max, Min

#### 3d. ✅ Complex Numbers - COMPLETE
- Added complex64/complex128 support to kernels via `num_complex::Complex`
- All binary ops (Add, Sub, Mul, Div, Pow, Mod, FloorDiv, Maximum, Minimum, Arctan2, Hypot, etc.)
- All unary ops (Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Tan, Sinh, Cosh, etc.)
- All reduce ops (Sum, Prod, Max, Min)

#### 3e. ✅ Comparison Ops - COMPLETE
- Added `CompareKernel<T>` trait returning bool
- Implemented for Gt, Lt, Ge, Le, Eq, Ne across all dtypes including complex
- Added `dispatch_compare_*` functions and `dispatch_compare_kernel`
- Added `map_compare` (contiguous) and `map_compare_strided` loop functions

#### 3f. Delete registry.rs - PARTIALLY BLOCKED
Registry still used for:
- Axis reductions (not migrated to kernel system)
- Bool-returning unary ops (Isnan, Isinf, Isfinite, Signbit) - handled via trait fallback

To fully delete registry.rs:
- Migrate axis reduce_axis_op to dispatch
- Remove registry from ufunc.rs

### Stream 4: Final Cleanup (optional)
- Slim DTypeOps trait (remove op implementations)
- Update macros.rs (remove op code generation)
- Update CLAUDE.md with new architecture

## Current State

**New orthogonal system**: 2009 lines
- dispatch.rs: 749 lines
- kernels/arithmetic.rs: 426 lines
- kernels/comparison.rs: 107 lines
- kernels/math.rs: 471 lines
- kernels/mod.rs: 37 lines
- loops/contiguous.rs: 117 lines
- loops/strided.rs: 90 lines
- loops/mod.rs: 12 lines

**Old registry (fallback)**: 1754 lines (reduced from 1840, still used for axis reductions, bitwise ops)

All 2191 tests pass. The new system handles ALL dtypes including complex:
- All 18 binary ops for f64, f32, i64, i32, i16, u64, u32, u16, u8, Complex<f64>, Complex<f32>
- All 31+ unary ops for float/int/complex types
- All 4 reduce ops for float/int/complex types
- All 6 comparison ops for all dtypes including complex

## Performance (release build, 1M f64 elements)

| Operation | vs NumPy | Notes |
|-----------|----------|-------|
| add | 1.03x | Good |
| mul | 1.10x | Good |
| div | 0.99x | Match |
| sqrt | 1.10x | Good |
| sin | 0.92x | Faster |
| sum | 1.33x | Needs SIMD |
| max | 3.13x | Needs SIMD |
| complex add | 1.40x | NumPy has SIMD |
| complex sqrt | 0.65x | Faster |

## Success Metrics

- ✅ Primary path now uses orthogonal kernel/loop architecture
- ✅ Adding new operation: 1 file (kernels/*.rs)
- ✅ Adding new dtype: impl kernel traits
- ✅ SIMD optimization: change `loops/contiguous.rs` once
- ✅ Complex numbers now use kernel/loop architecture (not registry)
- ✅ Comparison ops now use kernel/loop architecture
- ⚠️ registry.rs still exists for axis reductions

## Next Steps: Remove Dead Registry Code

The new dispatch system is called FIRST in ufunc.rs. Registry is only fallback.
Most registry loops are now dead code:

### Step 1: Remove dead binary registrations (~580 lines)
- `register_strided_binary!` macro and all invocations
- `register_arithmetic!`, `register_float_binary!`, `register_stream2_binary!`
- `register_f16_binary!`, `register_complex_loops!`
- Remove `lookup_binary`, `register_binary` from UFuncRegistry
- Remove `binary_loops` HashMap

### Step 2: Remove dead unary registrations (~120 lines)
- `register_strided_unary!` macro and invocations
- `register_float_unary!`, `register_f16_unary!`, `register_signed_int_unary!`
- Keep ONLY: Isnan, Isinf, Isfinite, Signbit (bool-returning, not in dispatch)
- Remove `lookup_unary`, `register_unary` from UFuncRegistry
- Remove `unary_loops` HashMap

### Step 3: Clean up ufunc.rs fallback paths
- Remove registry lookup code for binary/unary (dispatch handles it)
- Keep trait fallback for bool-returning ops and unsupported dtypes

## Future Work: Delete registry.rs entirely

After steps 1-3, registry.rs will only contain:
- Axis reductions (~400 lines) - need dispatch_reduce_axis
- Bitwise ops (~150 lines) - could migrate to kernel system
- Bool-returning unary fallback (~50 lines) - need UnaryKernel<In,Out>

1. ~~**Comparison ops**~~: ✅ DONE

2. **Axis reductions** (more complex):
   - Need per-axis output shape logic
   - Consider: extend dispatch_reduce with axis parameter

3. **Bool-returning unary ops** (Isnan, Isinf, Isfinite, Signbit):
   - Already handled by trait fallback (DTypeOps::unary_op)
   - Could add `UnaryKernel<In, Out>` for type-changing ops

## Architectural Limitations

Cases that don't fit cleanly in `Kernel<T> -> T` model:

| Case | Challenge | Solution |
|------|-----------|----------|
| Bool-returning ops | Output dtype differs | `UnaryKernel<In, Out>` trait |
| abs(complex) → float | Output dtype differs | Same as above |
| modf, frexp, divmod | Multiple outputs | Separate trait or special handling |
| out= parameter | Already partially supported | `map_binary_op_inplace` exists |
| Generalized ufuncs | Different signature | Separate system (matmul, dot) |

## Risk Mitigation

- Keep existing code working throughout
- Each phase produces working code
- Tests run after every change
- Can stop at any phase with partial benefit

## Key NumPy Insights (for reference)

NumPy's solution studied from `numpy/` checkout:

1. **ArrayMethod system** (`numpy/_core/src/umath/dispatching.cpp`):
   - Multiple dispatch on DType *classes* not instances
   - Slot-based: `NPY_METH_strided_loop`, `NPY_METH_contiguous_loop`
   - Cache lookup for repeated operations

2. **Multi-target SIMD** (`.dispatch.c.src` files):
   - Single source compiled multiple times with different CPU flags
   - Runtime CPU detection selects best implementation
   - Highway library wrapper (`numpy/_core/src/common/simd/`)

3. **Template code generation**:
   - `/**begin repeat` / `/**end repeat` generates code per dtype
   - Python preprocessor, not C macros

4. **Separation of concerns**:
   - Type resolution → Loop selection → Loop execution
   - Rumpy currently conflates these in `map_binary_op`

## Current Code Locations

- `src/ops/registry.rs` (1840 lines) - loop registrations, embeds layout checks
- `src/ops/ufunc.rs` - orchestration (map_unary_op, map_binary_op)
- `src/array/dtype/macros.rs` - generates DTypeOps impls with embedded ops
- `src/array/dtype/mod.rs` - DTypeOps trait, UnaryOp/BinaryOp/ReduceOp enums

## Example Current Pattern (to refactor)

```rust
// registry.rs line 265-296 - repeated for every (op, dtype) pair
macro_rules! register_strided_binary {
    ($reg:expr, $op:expr, $kind:expr, $T:ty, $f:expr) => {
        $reg.register_binary($op, TypeSignature::binary(...),
            |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
                if sa == itemsize && sb == itemsize && so == itemsize {
                    // Contiguous - REPEATED in every registration
                    for i in 0..n { out[i] = $f(a[i], b[i]); }
                } else {
                    // Strided - REPEATED in every registration
                    for _ in 0..n { ... pointer arithmetic ... }
                }
            });
    };
}
```

Target: Extract the contiguous/strided logic ONCE, parameterize by operation.
