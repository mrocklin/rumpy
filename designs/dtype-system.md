<!-- AI: Read this when working with dtypes -->

# DType System Design

## Motivation

Enable adding new dtypes with minimal code changes, including parametric types like `datetime[ns]` or `float64[meters]`.

## Architecture

**`DType` wraps `Arc<dyn DTypeOps>`**: This enables parametric types where the same dtype kind can have different parameters.

**`DTypeOps` trait**: Encapsulates all dtype-specific behavior:
- `kind()` - returns `DTypeKind` for equality/hashing
- `itemsize()`, `typestr()`, `format_char()`, `name()` - metadata
- Buffer-based ops (`add`, `mul`, `neg`, etc.) - each dtype uses its native type internally
- `to_pyobject()`, `from_pyobject()` - Python interop (the only universal interface)
- `promotion_priority()` - type promotion

**No universal Rust value type**: Operations work directly on buffers. Float64 casts to f64 internally, Complex128 works with (f64, f64) pairs, Decimal would use rust_decimal. Python interop (PyObject) is the only place all types converge.

**`DTypeKind` enum**: Used for equality, hashing, and pattern matching. Parametric types include their parameters in the enum variant.

**Hybrid dispatch (registry + trait)**: Operations check a global `UFuncRegistry` first for type-specific loops, falling back to `DTypeOps` trait methods. This enables:
- Optimized loops for common types without touching trait code
- Type-specific behavior (e.g., bitwise only for ints)
- New dtypes work immediately via trait fallback

**NumPy-compatible type promotion**: `promote_dtype()` follows NumPy's safe-casting rules (e.g., int64+float32→float64).

## Adding a Simple DType

1. **Create dtype file**: `src/array/dtype/newtype.rs` with struct + `DTypeOps` impl
2. **Register in dtype module** (`src/array/dtype/mod.rs`):
   - Add `mod newtype;` and `use newtype::NewTypeOps;`
   - Add variant to `DTypeKind` enum
   - Add constructor `DType::newtype()`
   - Add string parsing in `DType::from_str()` (e.g., `"newtype" | "<n8"`)
   - Update `promote_dtype()` if the type participates in promotion
3. **Add registry loops** (`src/ops/registry.rs`):
   - Binary ops (Add, Sub, Mul, Div) for same-type operations
   - Unary ops (Neg, Abs, etc.) as appropriate
   - Reduce ops (Sum, Prod, Max, Min) as appropriate
4. **Update ops dispatch** (`src/ops/mod.rs`):
   - Add to transcendental checks if it's a float-like type
   - Add to complex checks if it's a complex type
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

## Existing Macros

For types that share similar structure, use existing macros in `src/ops/registry.rs`:
- `register_complex_loops!($reg, $kind, $T)` - registers all binary/unary/reduce ops for complex types (used for Complex64 and Complex128)
- `register_strided_unary!` - registers unary ops with contiguous fast path
- `register_float_reduce!`, `register_int_reduce!` - reduce ops for numeric types

## Key Files

- `src/array/dtype/mod.rs` - `DType`, `DTypeKind`, `DTypeOps` trait, `promote_dtype()`
- `src/array/dtype/*.rs` - dtype implementations (float64, int64, uint8, datetime64, complex64, etc.)
- `src/ops/registry.rs` - `UFuncRegistry` for type-specific inner loops
- `src/ops/mod.rs` - `map_binary_op`, `map_unary_op` with registry dispatch
- `src/python/mod.rs` - Python typestr parsing in `dtype_from_typestr()`
