# Kernel/Dispatch Architecture

Orthogonal separation of Operations, Layouts, and DTypes for ufuncs.

## Core Principle

Three dimensions that vary independently:
```
Operation  (add, sqrt, sum, bitwise_and)  → kernels/*.rs
Layout     (contiguous, strided)          → loops/*.rs
DType      (f64, i32, complex, bool)      → impl Kernel<T> for Op
```

Adding a new operation: implement kernel traits in `kernels/`.
Adding a new dtype: add `impl Kernel<T>` for existing ops.
Optimizing layout: change `loops/` once, all ops benefit.

## File Structure

```
src/ops/
├── kernels/
│   ├── mod.rs         # Traits: BinaryKernel, UnaryKernel, ReduceKernel, CompareKernel
│   ├── arithmetic.rs  # Add, Sub, Mul, Div, Sum, Prod, Max, Min, Pow, etc.
│   ├── bitwise.rs     # And, Or, Xor, LeftShift, RightShift, Not
│   ├── comparison.rs  # Gt, Lt, Ge, Le, Eq, Ne
│   └── math.rs        # Sqrt, Exp, Log, Sin, Cos, etc.
├── loops/
│   ├── mod.rs         # Re-exports
│   ├── contiguous.rs  # Slice-based loops (SIMD-friendly)
│   └── strided.rs     # Pointer arithmetic loops
├── dispatch.rs        # Type resolution + layout detection → kernel + loop
└── ufunc.rs           # Public API: map_unary_op, map_binary_op, reduce_axis_op
```

## Kernel Traits

```rust
pub trait BinaryKernel<T>: Copy {
    fn apply(a: T, b: T) -> T;
}

pub trait UnaryKernel<T>: Copy {
    fn apply(v: T) -> T;
}

pub trait ReduceKernel<T>: Copy {
    fn init() -> T;
    fn combine(acc: T, v: T) -> T;
}

pub trait CompareKernel<T>: Copy {
    fn apply(a: T, b: T) -> bool;
}
```

Kernels are zero-sized types. `K::apply(a, b)` monomorphizes to tight code per (kernel, dtype) pair.

## Dispatch Flow

```
map_binary_op(a, b, BinaryOp::Add)
    → dispatch::dispatch_binary_add(a, b, out_shape)
        → match dtype:
            Float64 → dispatch_binary_typed::<f64, Add>(...)
                → if contiguous: loops::map_binary(slice_a, slice_b, out, Add)
                   else:         loops::map_binary_strided(ptr_a, stride_a, ..., Add)
```

## Why Traits Over Closures

Closures prevent monomorphization. With traits:
```rust
for i in 0..n { out[i] = K::apply(a[i], b[i]); }
```
Compiles to tight scalar or SIMD code. The compiler sees the exact operation.

## Loop Strategies

**Contiguous** (`loops/contiguous.rs`):
- Operates on slices
- LLVM can auto-vectorize
- 4-accumulator pattern for reductions (ILP)

**Strided** (`loops/strided.rs`):
- Pointer arithmetic with byte offsets
- No SIMD benefit (irregular access)
- Handles views, transposes, broadcasts

## Fallback Chain

1. **Dispatch** (kernel/loop) - handles f64, f32, f16, i64, i32, i16, u64, u32, u16, u8, complex128, complex64, bool (for bitwise)
2. **Registry** - Bool reduce loops only
3. **Trait dispatch** - DateTime64, any future types

## Extending the System

**New binary op** (e.g., `lcm`):
```rust
// kernels/arithmetic.rs
pub struct Lcm;
impl BinaryKernel<i64> for Lcm {
    fn apply(a: i64, b: i64) -> i64 { /* ... */ }
}
// Similar for i32, u64, etc.

// dispatch.rs
pub fn dispatch_binary_lcm(...) { dispatch_binary_kernel(..., Lcm) }

// ufunc.rs
BinaryOp::Lcm => dispatch::dispatch_binary_lcm(a, b, out_shape),
```

**New dtype**: Add `impl BinaryKernel<NewType> for Add` etc., then add match arm in dispatch.
