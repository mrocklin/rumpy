# Adding Operations

Guide for adding new ufunc operations (unary, binary, reduce).

## Files to Modify

1. **`src/array/dtype/mod.rs`** - Add to enum (`UnaryOp`, `BinaryOp`, or `ReduceOp`)

2. **`src/array/dtype/*.rs`** - Implement in each dtype's `DTypeOps`:
   - `float64.rs`, `float32.rs`, `float16.rs` - floating point
   - `int64.rs`, `int32.rs`, `int16.rs` - signed integers
   - `uint64.rs`, `uint32.rs`, `uint16.rs`, `uint8.rs` - unsigned integers
   - `bool.rs`, `complex128.rs`, `datetime64.rs` - special types

3. **`src/ops/registry.rs`** - Register fast loops (optional but recommended):
   - Use `register_strided_binary!` / `register_strided_unary!` macros
   - Or existing group macros: `register_arithmetic!`, `register_float_unary!`

4. **`src/ops/mod.rs`** - Update mixed-type fallback match arms (for binary ops)

5. **`src/python/pyarray.rs`** - Add Python bindings (`__pow__`, `fn sqrt`, etc.)

6. **`tests/test_*.py`** - Add tests comparing against NumPy

## Example: Adding `BinaryOp::Pow`

```rust
// 1. src/array/dtype/mod.rs
pub enum BinaryOp {
    Add, Sub, Mul, Div,
    Pow,  // <-- add here
}

// 2. src/array/dtype/float64.rs (and other dtypes)
BinaryOp::Pow => av.powf(bv),

// 3. src/ops/registry.rs - add to register_arithmetic! or individually
register_strided_binary!(reg, BinaryOp::Pow, DTypeKind::Float64, f64, |a: f64, b: f64| a.powf(b));

// 4. src/python/pyarray.rs (note: __pow__ requires modulo param)
fn __pow__(&self, other: &Bound<'_, PyAny>, _modulo: &Bound<'_, PyAny>) -> PyResult<Self> {
    binary_op_dispatch(&self.inner, other, BinaryOp::Pow)
}
```

## Notes

- Registry loops are optional; fallback uses `DTypeOps` trait methods
- Registry provides strided fast path with SIMD auto-vectorization
- Not all ops make sense for all types (e.g., trig for integers, pow for bool)
