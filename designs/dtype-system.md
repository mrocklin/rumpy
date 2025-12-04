<!-- AI: Read this when working with dtypes -->

# DType System Design

## Motivation

Enable adding new dtypes with minimal code changes, including parametric types like `datetime[ns]` or `float64[meters]`.

## Architecture

**`DType` wraps `Arc<dyn DTypeOps>`**: This enables parametric types where the same dtype kind can have different parameters.

**`DTypeOps` trait**: Encapsulates all dtype-specific behavior:
- `kind()` - returns `DTypeKind` for equality/hashing
- `itemsize()`, `typestr()`, `format_char()`, `name()` - metadata
- `read_element()`, `write_element()` - pointer casting
- `zero_value()`, `one_value()` - value creation
- `promotion_priority()` - type promotion

**`DTypeKind` enum**: Used for equality, hashing, and pattern matching. Parametric types include their parameters in the enum variant.

## Adding a Simple DType

1. Create `src/array/dtype/newtype.rs` with struct + `DTypeOps` impl
2. Add variant to `DTypeKind` enum
3. Add constructor `DType::newtype()` in `dtype/mod.rs`

## Adding a Parametric DType

```rust
// 1. Add to DTypeKind
enum DTypeKind {
    // ...
    DateTime(TimeUnit),
}

// 2. Create struct with parameter
pub struct DateTimeOps {
    pub unit: TimeUnit,
}

impl DTypeOps for DateTimeOps {
    fn kind(&self) -> DTypeKind { DTypeKind::DateTime(self.unit) }
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
    pub fn datetime(unit: TimeUnit) -> Self {
        DType::new(DateTimeOps { unit })
    }
}
```

## Key Files

- `src/array/dtype/mod.rs` - `DType`, `DTypeKind`, `DTypeOps` trait
- `src/array/dtype/{float64,float32,int64,int32,bool}.rs` - dtype implementations
