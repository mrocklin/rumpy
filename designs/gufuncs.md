# Gufunc Design

## Core Principle

Gufuncs (generalized universal functions) extend ufuncs to operate on **sub-arrays** rather than scalars. They enable operations like matrix multiplication that have "core dimensions" while still supporting broadcasting over "loop dimensions".

## Signature Notation

A signature describes core dimensions: `(m,n),(n,p)->(m,p)`

- **Core dimensions**: Named dims at the END of each array's shape
- **Loop dimensions**: All preceding dims, broadcast together
- **Dimension binding**: Same name (e.g., `n`) must have same size across arrays

Example for matmul `[3,2,4] @ [3,4,5] → [3,2,5]`:
- Loop shape: `[3]`
- Input 1 core: `(2,4)` binds `m=2, n=4`
- Input 2 core: `(4,5)` binds `n=4, p=5` (n must match)
- Output core: `(2,5)` from `m=2, p=5`

## Architecture

```
GufuncSignature (parsed)
        ↓
GufuncDims::resolve(sig, inputs) → resolved sizes
        ↓
Loop over broadcast dimensions:
    - Extract core sub-arrays via view_with (O(1))
    - Call kernel(inputs, outputs)
        ↓
Return output arrays
```

## Kernel Interface

Trait-based for extensibility (BLAS integration):

```rust
pub trait GufuncKernel: Send + Sync {
    fn signature(&self) -> &GufuncSignature;
    fn call(&self, inputs: &[RumpyArray], outputs: &mut [RumpyArray]);
}
```

## Key Design Decisions

1. **Runtime signature parsing** - Flexible, matches NumPy's approach
2. **View-based sub-arrays** - Zero-copy, leverages existing Arc<ArrayBuffer>
3. **Trait kernels** - Allows swapping simple Rust impl for BLAS
4. **Separate dims resolution** - Clear error handling for shape mismatches

## Adding New Gufuncs

1. Define kernel struct implementing `GufuncKernel`
2. Return signature from `signature()` method
3. Implement core operation in `call()` method
4. Create wrapper function that calls `gufunc_call()`
