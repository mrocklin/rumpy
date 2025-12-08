<!-- AI: Read this when working with dtypes -->

# DType System Design

## Motivation

Enable adding new dtypes with minimal code changes, including parametric types like `datetime[ns]` or `float64[meters]`.

## Architecture

**`DType` wraps `Arc<dyn DTypeOps>`**: This enables parametric types where the same dtype kind can have different parameters.

**`DTypeOps` trait**: Encapsulates all dtype-specific behavior:
- `kind()` - returns `DTypeKind` for equality/hashing
- `itemsize()`, `typestr()`, `format_char()`, `name()` - metadata
- Buffer-based ops (`add`, `mul`, `neg`, etc.) - fallback implementations
- `to_pyobject()`, `from_pyobject()` - Python interop (the only universal interface)
- `promotion_priority()` - type promotion

**No universal Rust value type**: Operations work directly on buffers. Float64 casts to f64 internally, Complex128 works with (f64, f64) pairs, Decimal would use rust_decimal. Python interop (PyObject) is the only place all types converge.

**`DTypeKind` enum**: Used for equality, hashing, and pattern matching. Parametric types include their parameters in the enum variant.

**Kernel/dispatch for operations**: Primary path uses kernel/dispatch system (see `designs/kernel-dispatch.md`). `DTypeOps` trait provides fallback for unsupported types.

**NumPy-compatible type promotion**: `promote_dtype()` follows NumPy's safe-casting rules (e.g., int64+float32→float64).

## Adding a Simple DType

1. **Create dtype file**: `src/array/dtype/newtype.rs` with struct + `DTypeOps` impl
2. **Register in dtype module** (`src/array/dtype/mod.rs`):
   - Add `mod newtype;` and `use newtype::NewTypeOps;`
   - Add variant to `DTypeKind` enum
   - Add constructor `DType::newtype()`
   - Add string parsing in `DType::from_str()` (e.g., `"newtype" | "<n8"`)
   - Update `promote_dtype()` if the type participates in promotion
3. **Add kernel impls** (`src/ops/kernels/*.rs`):
   - `impl BinaryKernel<NewType> for Add { ... }` etc.
   - `impl UnaryKernel<NewType> for Sqrt { ... }` etc.
   - `impl ReduceKernel<NewType> for Sum { ... }` etc.
4. **Add dispatch arms** (`src/ops/dispatch.rs`):
   - Add `DTypeKind::NewType` cases to dispatch functions
5. **Add Python bindings** (`src/python/mod.rs`):
   - Add typestr parsing (e.g., `('n', 8) => Ok(DType::newtype())`)
6. **Add tests**: Create parametrized tests in `tests/test_*.py` covering the new dtype

## Adding a Parametric DType

```rust
// 1. Add to DTypeKind
enum DTypeKind {
    // ...
    DateTime64(TimeUnit),
}

// 2. Create struct with parameter
struct DateTime64Ops {
    unit: TimeUnit,
}

impl DTypeOps for DateTime64Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::DateTime64(self.unit) }
    fn itemsize(&self) -> usize { 8 }
    fn name(&self) -> &'static str {
        match self.unit {
            TimeUnit::Nanoseconds => "datetime64[ns]",
            // ...
        }
    }
    // ...
}

// 3. Add constructor
impl DType {
    pub fn datetime64(unit: TimeUnit) -> Self {
        DType::new(DateTime64Ops { unit })
    }
}
```

## Key Files

- `src/array/dtype/mod.rs` - `DType`, `DTypeKind`, `DTypeOps` trait, `promote_dtype()`
- `src/array/dtype/*.rs` - dtype implementations via macros (floats.rs, integers.rs, complex64.rs, etc.)
- `src/ops/kernels/*.rs` - kernel implementations per dtype
- `src/ops/dispatch.rs` - type resolution and layout selection
- `src/ops/ufunc.rs` - `map_binary_op`, `map_unary_op` with dispatch
- `src/python/mod.rs` - Python typestr parsing in `dtype_from_typestr()`
