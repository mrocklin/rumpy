# Adding Operations

**See `designs/kernel-dispatch.md`** for the current architecture.

## Quick Reference

1. **Add kernel** in `src/ops/kernels/*.rs`:
   ```rust
   pub struct MyOp;
   impl BinaryKernel<f64> for MyOp { fn apply(a: f64, b: f64) -> f64 { ... } }
   ```

2. **Add dispatch** in `src/ops/dispatch.rs`:
   ```rust
   pub fn dispatch_binary_myop(...) { dispatch_binary_kernel(..., MyOp) }
   ```

3. **Wire up** in `src/ops/ufunc.rs`:
   ```rust
   BinaryOp::MyOp => dispatch::dispatch_binary_myop(a, b, out_shape),
   ```

4. **Python bindings** in `src/python/pyarray/*.rs`

5. **Tests** in `tests/test_*.py`

## Why Not Registry?

The old registry (`src/ops/registry.rs`) embedded operation logic inside closures, preventing monomorphization. The kernel/dispatch system separates:

- **Operations** (kernels) - pure math, no memory concerns
- **Layouts** (loops) - contiguous vs strided traversal
- **DTypes** (dispatch) - type resolution and selection

This enables SIMD optimization in one place (`loops/contiguous.rs`) for all operations.
