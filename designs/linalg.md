# Linear Algebra Design

Rumpy uses **faer** for optimized linear algebra operations.

## Why faer

- **Pure Rust**: No external BLAS linking, simpler distribution
- **Performance**: Competitive with OpenBLAS, slightly behind MKL
- **Zero-copy views**: `MatRef`/`MatMut` wrap existing memory via `from_raw_parts`
- **Flexible strides**: Supports arbitrary row/col strides (not just column-major)

## Integration Pattern

```rust
use faer::mat;

// Convert byte strides to element strides
let elem_size = std::mem::size_of::<f64>() as isize;
let row_stride = arr.strides()[0] / elem_size;
let col_stride = arr.strides()[1] / elem_size;

// Create zero-copy view
let mat_ref = unsafe {
    mat::from_raw_parts::<f64, usize, usize>(
        arr.data_ptr() as *const f64,
        nrows, ncols,
        row_stride, col_stride,
    )
};
```

## Key APIs

- `mat::from_raw_parts` - immutable view from pointer + strides
- `mat::from_raw_parts_mut` - mutable view
- `faer::linalg::matmul::matmul` - GEMM operation
- Decompositions: LU, QR, Cholesky, SVD, eigendecomposition

## Trade-offs

| Approach | Binary Size | Performance | Complexity |
|----------|-------------|-------------|------------|
| Triple loop | +0 | Slow | Simple |
| faer | +12MB | ~OpenBLAS | Medium |
| MKL (feature) | +0 (dynamic) | Best | Complex linking |

## Stride Considerations

- RumpyArray uses **byte strides** (like NumPy)
- faer uses **element strides**
- Always convert: `elem_stride = byte_stride / sizeof(T)`
- faer handles non-contiguous memory correctly

## Implemented

- `solve` - linear system via LU
- `qr` - QR decomposition (thin)
- `svd` - SVD decomposition (thin)
- `det`, `trace`, `norm` (Frobenius)
- `inv` - matrix inverse via LU
- `eigh` - symmetric eigendecomposition
- `diag` - extract/create diagonal

## Future Work

- `eig` - general eigendecomposition (needs complex dtype)
- `lstsq`, `pinv` - least squares, pseudo-inverse
- More norm types
