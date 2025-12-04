<!-- AI: Read this when working with dtypes -->

# DType System Design

## Motivation

Enable adding new dtypes (uint8, complex64, datetime, etc.) with minimal code changes.

## Architecture

**Trait-based design**: Each dtype is a unit struct implementing `DTypeOps`. The trait encapsulates all dtype-specific behavior:
- Element read/write (pointer casting)
- Metadata (itemsize, typestr)
- Value creation (zero, one)
- Type promotion (priority-based)

**Lightweight enum**: `DType` enum remains for storage/pattern matching but delegates all behavior to `dtype.ops()`.

## Adding a New DType

1. Create `src/array/dtype/newtype.rs` with struct + `DTypeOps` impl
2. Add variant to `DType` enum in `dtype/mod.rs`
3. Add one match arm in `DType::ops()`

All constructors, operations, and reductions automatically work via the trait.

## Key Files

- `src/array/dtype/mod.rs` - `DTypeOps` trait, `DType` enum, `promote_dtype()`
- `src/array/dtype/{float64,float32,int64,int32,bool}.rs` - dtype implementations

## Future: Parametric Types

For datetime[ns] or float64[meters], enum variants can hold parameters:
```rust
enum DType {
    DateTime(TimeUnit),
}
```
Or use `Arc<dyn DTypeOps>` for full runtime flexibility.
