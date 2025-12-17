# Rumpy

A NumPy-compatible array library implemented in Rust with Python bindings via PyO3.

## Why Rumpy?

Rumpy explores what a NumPy implementation looks like built from scratch in Rust:

- **Memory safety** - Rust's ownership model prevents buffer overflows and use-after-free
- **Clean architecture** - Orthogonal separation of dtypes, operations, and memory layouts
- **Modern tooling** - Leverages Rust ecosystem (faer for linear algebra, rustfft for FFT)
- **NumPy compatibility** - Drop-in replacement for most NumPy operations

## Why not Rumpy?

This is mostly a fun project to play with AI.  It is not actively maintained.
Please use Numpy instead.  Numpy is managed by actual humans.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/rumpy.git
cd rumpy

# Create virtual environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install pytest numpy

# Build the extension (compiles Rust and installs into venv)
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
```

## Usage

```python
import rumpy as rp

# Array creation
a = rp.zeros((3, 4), dtype="float64")
b = rp.arange(12).reshape((3, 4))
c = rp.linspace(0, 1, 100)

# Arithmetic (supports broadcasting)
d = a + b * 2
e = rp.sqrt(b) + rp.sin(c[:, None])

# Reductions
total = b.sum()
row_means = b.mean(axis=1)
col_max = b.max(axis=0)

# Linear algebra
x = rp.linalg.solve(A, b)
u, s, vh = rp.linalg.svd(A)

# Random numbers
rng = rp.random.default_rng(42)
samples = rng.normal(0, 1, size=(1000,))

# FFT
spectrum = rp.fft.fft(signal)

# NumPy interop (zero-copy when possible)
import numpy as np
n = np.asarray(b)  # rumpy -> numpy
r = rp.asarray(n)  # numpy -> rumpy
```

## Supported Features

| Category | Functions |
|----------|-----------|
| Creation | `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `eye`, `*_like` |
| Math | `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `arcsin`, `abs`, `sign`, ... |
| Arithmetic | `+`, `-`, `*`, `/`, `**`, `//`, `%`, `@` (matmul) |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=`, `isclose`, `allclose` |
| Reductions | `sum`, `mean`, `std`, `var`, `min`, `max`, `argmin`, `argmax`, `all`, `any` |
| Shape | `reshape`, `transpose`, `stack`, `concatenate`, `split`, `flip`, `squeeze` |
| Indexing | Slicing, `take`, `put`, `searchsorted`, `where`, `compress` |
| Sorting | `sort`, `argsort`, `partition`, `unique`, `lexsort` |
| Linear algebra | `matmul`, `dot`, `solve`, `inv`, `det`, `eig`, `svd`, `qr`, `lstsq` |
| FFT | `fft`, `ifft`, `fft2`, `ifft2`, `rfft`, `irfft` |
| Random | `random`, `integers`, `normal`, `uniform`, `choice`, `shuffle` |
| Statistics | `histogram`, `cov`, `corrcoef`, `median`, `percentile` |
| Polynomials | `polyfit`, `polyval`, `polyder`, `polyint`, `roots` |
| Set operations | `unique`, `isin`, `intersect1d`, `union1d`, `setdiff1d` |

## DTypes

Supported: `bool`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float32`, `float64`, `complex64`, `complex128`

## Repository Structure

```
src/
├── array/           # Core array type, buffer management, dtype system
├── ops/             # Operations: kernels, dispatch, loops
│   ├── kernels/     # Pure operation definitions (add, mul, sqrt, ...)
│   ├── loops/       # Memory traversal (contiguous SIMD, strided)
│   └── ...          # Domain-specific ops (linalg, fft, statistics)
├── python/          # PyO3 bindings organized by category
└── random/          # Random number generation

tests/               # pytest tests (every test compares against numpy)
designs/             # Architecture documentation
plans/               # Development tracking
```

### Key Architecture

**Kernel/Dispatch separation** - Operations are defined as zero-sized kernel types, separate from memory traversal strategies. This allows SIMD optimizations in one place (contiguous loops) that benefit all operations.

**Arc-wrapped buffers** - Views share ownership via `Arc<ArrayBuffer>`, enabling zero-copy slicing and NumPy interop.

**Trait-based dtypes** - `DType(Arc<dyn DTypeOps>)` with macro-generated implementations for each concrete type.

## Known Differences from NumPy

See `designs/deviations.md` for details:

- Rounding uses "half away from zero" (Rust default) vs NumPy's "half to even"
- `inner()` uses gufunc broadcasting instead of Cartesian product
- Random integers use Lemire's algorithm (uniform but not bit-identical to NumPy)
- BLAS acceleration only on macOS (via Accelerate); other platforms use pure Rust

## Development

```bash
# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_reductions.py -v

# Rebuild after Rust changes
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
```
