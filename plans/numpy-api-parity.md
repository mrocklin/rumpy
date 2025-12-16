# NumPy API Parity Plan

**Design Ref**: `designs/adding-operations.md`, `designs/ufuncs.md`

## Overview

This plan covers ~130 missing NumPy functions organized into parallelizable work streams.
Each stream is independent and can be worked on by a separate agent.

---

## Stream 1: Math Operations (Unary) ✅ COMPLETE

Simple element-wise operations using `map_unary`. Pattern: add to `UnaryOp` enum,
implement in dtype files, add Python binding, test.

### Tier 1 - Common
- [x] `square` - x²
- [x] `negative` - -x
- [x] `positive` - +x (identity for numeric)
- [x] `reciprocal` - 1/x
- [x] `abs`/`absolute` - already have abs, add alias

### Tier 2 - Precision Variants
- [x] `exp2` - 2^x
- [x] `expm1` - e^x - 1 (precise for small x)
- [x] `log1p` - log(1 + x) (precise for small x)
- [x] `cbrt` - cube root

### Tier 3 - Rounding
- [x] `trunc` - truncate toward zero
- [x] `rint` - round to nearest integer
- [x] `fix` - round toward zero (alias for trunc)

### Tier 4 - Inverse Hyperbolic
- [x] `arcsinh` - inverse sinh
- [x] `arccosh` - inverse cosh
- [x] `arctanh` - inverse tanh

### Tier 5 - Misc
- [x] `signbit` - true if sign bit set
- [x] `nan_to_num` - replace nan/inf with numbers

**Files**: `src/array/dtype/mod.rs`, `src/array/dtype/*.rs`, `src/ops/registry.rs`,
`src/python/pyarray.rs`, `tests/test_math.py`

---

## Stream 2: Math Operations (Binary) ✅ COMPLETE

Element-wise binary ops using `map_binary` with broadcasting.

### Tier 1 - Essential
- [x] `arctan2` - two-argument arctangent
- [x] `hypot` - sqrt(x² + y²)
- [x] `mod` - alias for remainder
- [x] `fmax` - element-wise max ignoring NaN
- [x] `fmin` - element-wise min ignoring NaN
- [x] `copysign` - copy sign from y to x

### Tier 2 - Precision
- [x] `logaddexp` - log(exp(x) + exp(y))
- [x] `logaddexp2` - log2(2^x + 2^y)
- [x] `nextafter` - next floating point value toward y

### Tier 3 - Angular
- [x] `deg2rad` / `radians` - degrees to radians
- [x] `rad2deg` / `degrees` - radians to degrees

**Files**: Same as Stream 1 plus `src/ops/mod.rs` for mixed-type fallbacks.

---

## Stream 3: Comparison Operations ✅ COMPLETE

Return boolean arrays. Use `map_binary` with bool output.

### Core Comparisons
- [x] `equal` - x == y (element-wise)
- [x] `not_equal` - x != y
- [x] `less` - x < y
- [x] `less_equal` - x <= y
- [x] `greater` - x > y
- [x] `greater_equal` - x >= y

### Approximate Comparisons
- [x] `isclose` - element-wise approximate equality
- [x] `allclose` - all elements approximately equal (returns scalar)
- [x] `array_equal` - arrays have same shape and elements

**Note**: The `__lt__`, `__eq__` etc. operators may already exist. These are the
function forms that can take `out` parameter and work with `where`.

**Files**: `src/array/dtype/mod.rs` (add ComparisonOp enum), `src/ops/comparison.rs` (new),
`src/python/mod.rs`, `tests/test_comparison.py`

---

## Stream 4: Logical Operations ✅ COMPLETE

Boolean logic, both element-wise and reduction.

### Element-wise
- [x] `logical_and` - element-wise AND
- [x] `logical_or` - element-wise OR
- [x] `logical_not` - element-wise NOT
- [x] `logical_xor` - element-wise XOR

**Files**: `src/ops/comparison.rs`, `src/python/mod.rs`

---

## Stream 5: Bitwise Operations ✅ COMPLETE

Integer-only bit manipulation.

- [x] `bitwise_and` - &
- [x] `bitwise_or` - |
- [x] `bitwise_xor` - ^
- [x] `bitwise_not` / `invert` - ~
- [x] `left_shift` - <<
- [x] `right_shift` - >>

**Files**: `src/ops/bitwise.rs`, `src/python/mod.rs`, `tests/test_bitwise.py`

---

## Stream 6: Reductions (NaN-aware) ✅ COMPLETE

Same as existing reductions but ignore NaN values.

- [x] `nansum` - sum ignoring NaN
- [x] `nanprod` - product ignoring NaN
- [x] `nanmean` - mean ignoring NaN
- [x] `nanstd` - std ignoring NaN
- [x] `nanvar` - var ignoring NaN
- [x] `nanmin` - min ignoring NaN
- [x] `nanmax` - max ignoring NaN
- [x] `nanargmin` - argmin ignoring NaN
- [x] `nanargmax` - argmax ignoring NaN

**Pattern**: Wrap existing reduction with NaN check in accumulator.

**Files**: `src/ops/mod.rs`, `src/python/mod.rs`, `tests/test_nan.py`

---

## Stream 7: Statistical Operations ✅ COMPLETE

- [x] `median` - median value (requires sorting)
- [x] `average` - weighted average
- [x] `ptp` - peak-to-peak (max - min)
- [x] `histogram` - compute histogram
- [x] `corrcoef` - correlation coefficient matrix
- [x] `cov` - covariance matrix

**Files**: `src/ops/statistics.rs`, `src/python/mod.rs`, `tests/test_statistics.py`

---

## Stream 8: Array Creation ✅ COMPLETE

### Tier 1 - Common
- [x] `full_like` - like full but match shape/dtype of input
- [x] `identity` - identity matrix
- [x] `logspace` - logarithmically spaced values
- [x] `geomspace` - geometrically spaced values

### Tier 2 - Triangular
- [x] `tri` - triangular matrix of ones
- [x] `tril` - lower triangle
- [x] `triu` - upper triangle
- [x] `diagflat` - create diagonal matrix from flat input

### Tier 3 - Advanced
- [x] `meshgrid` - coordinate matrices from vectors
- [x] `indices` - grid of indices
- [x] `fromfunction` - construct from function

**Files**: `src/array/mod.rs`, `src/python/mod.rs`, `tests/test_creation.py`

---

## Stream 9: Shape Manipulation (Module-level) ✅ COMPLETE

Module-level versions of array methods + new functions.

### Tier 1 - Aliases
- [x] `reshape` - module-level reshape
- [x] `ravel` - flatten to 1D (return view if possible)
- [x] `flatten` - flatten to 1D (always copy)
- [x] `transpose` - module-level transpose

### Tier 2 - Dimension Manipulation
- [x] `atleast_1d` - ensure at least 1D
- [x] `atleast_2d` - ensure at least 2D
- [x] `atleast_3d` - ensure at least 3D
- [x] `moveaxis` - move axis to new position
- [x] `rollaxis` - roll axis backward

### Tier 3 - Broadcasting
- [x] `broadcast_to` - broadcast array to shape
- [x] `broadcast_arrays` - broadcast multiple arrays together

**Files**: `src/array/mod.rs`, `src/python/mod.rs`, `tests/test_shape.py`

---

## Stream 10: Indexing Operations ✅ COMPLETE

### Tier 1 - Selection
- [x] `take` - take elements along axis
- [x] `take_along_axis` - take using index array along axis
- [x] `compress` - select elements using boolean mask along axis

### Tier 2 - Search
- [x] `searchsorted` - find insertion points for sorted array
- [x] `argwhere` - indices where condition is true
- [x] `flatnonzero` - indices of non-zero elements in flattened array

### Tier 3 - Modification
- [x] `put` - replace elements at indices
- [x] `put_along_axis` - put values using index array
- [x] `choose` - construct array from index array and choices

**Files**: `src/ops/indexing.rs`, `src/python/mod.rs`, `tests/test_indexing_ops.py`

---

## Stream 11: Array Manipulation ✅ COMPLETE

### Tier 1 - Splitting
- [x] `hsplit` - horizontal split (along axis 1) - returns views
- [x] `vsplit` - vertical split (along axis 0) - returns views
- [x] `dsplit` - depth split (along axis 2) - returns views

### Tier 2 - Stacking
- [x] `column_stack` - stack 1D as columns
- [x] `row_stack` - alias for vstack
- [x] `dstack` - stack along third axis

### Tier 3 - Repetition
- [x] `repeat` - repeat elements
- [x] `tile` - tile array (optimized memcpy for 1D)

### Tier 4 - Modification
- [x] `append` - append to array
- [x] `insert` - insert into array
- [x] `delete` - delete from array
- [x] `pad` - pad array (constant, edge modes)

### Tier 5 - Rotation
- [x] `roll` - roll elements along axis (optimized memcpy for flat)
- [x] `rot90` - rotate 90 degrees (returns views, O(1))

**Performance Notes**:
- `rot90`, `hsplit`, `vsplit`, `dsplit` return views (zero-copy, O(1))
- `tile` 1D and `roll` flat use memcpy for near-numpy performance
- Multi-dimensional operations still use element-by-element (room for optimization)

**Files**: `src/array/mod.rs`, `src/python/mod.rs`, `tests/test_manipulation.py`

---

## Stream 12: Set Operations ✓

- [x] `isin` - test if elements are in test array
- [x] `in1d` - test if elements are in 1D array (deprecated, alias to isin)
- [x] `intersect1d` - intersection of sorted arrays
- [x] `union1d` - union of arrays
- [x] `setdiff1d` - set difference
- [x] `setxor1d` - set symmetric difference

**Files**: `src/ops/set_ops.rs`, `src/python/mod.rs`, `tests/test_set.py`

---

## Stream 13: Sorting/Searching Advanced ✅ COMPLETE

- [x] `partition` - partial sort
- [x] `argpartition` - indices for partial sort
- [x] `lexsort` - indirect sort using multiple keys

**Files**: `src/ops/mod.rs`, `src/python/mod.rs`, `tests/test_sorting.py`

---

## Stream 14: Linear Algebra Extensions ✅ COMPLETE

### linalg submodule
- [x] `eig` - general eigenvalue decomposition
- [x] `eigvals` - eigenvalues only
- [x] `lstsq` - least squares solution
- [x] `pinv` - Moore-Penrose pseudo-inverse
- [x] `matrix_rank` - matrix rank
- [x] `cond` - condition number
- [x] `slogdet` - sign and log of determinant

### Module-level
- [x] `tensordot` - tensor dot product
- [x] `vdot` - vector dot product (conjugate first arg)
- [x] `kron` - Kronecker product
- [x] `cross` - cross product

**Files**: `src/ops/linalg.rs`, `src/python/linalg.rs`, `tests/test_linalg.py`

---

## Stream 15: Random Extensions ✅ COMPLETE

### Generator methods
- [x] `permutation` - random permutation
- [x] `shuffle` - shuffle in place
- [x] `beta` - beta distribution
- [x] `gamma` - gamma distribution
- [x] `poisson` - Poisson distribution
- [x] `binomial` - binomial distribution
- [x] `chisquare` - chi-square distribution
- [x] `multivariate_normal` - multivariate normal

**Files**: `src/random/*.rs`, `src/python/random.rs`, `tests/test_random.py`

---

## Stream 16: Numerical Operations ✅ COMPLETE

- [x] `gradient` - numerical gradient
- [x] `trapezoid` - trapezoidal integration
- [x] `interp` - 1D linear interpolation
- [x] `convolve` - already have, verified edge cases
- [x] `correlate` - cross-correlation

**Files**: `src/ops/numerical.rs` (new), `src/python/mod.rs`, `tests/test_numerical.py`

---

## Stream 17: Polynomial Operations ✓ COMPLETE

- [x] `polyfit` - polynomial curve fitting
- [x] `polyval` - evaluate polynomial
- [x] `polyder` - polynomial derivative
- [x] `polyint` - polynomial integral
- [x] `roots` - polynomial roots

**Files**: `src/ops/poly.rs`, `src/python/mod.rs`, `tests/test_poly.py`

---

## Stream 18: ndarray Methods ✅ COMPLETE

Methods missing from the array class itself.

### Tier 1 - Index operations
- [x] `nonzero()` method - indices of non-zero elements
- [x] `argsort()` method
- [x] `sort()` method - in-place sort
- [x] `searchsorted()` method

### Tier 2 - Element operations
- [x] `repeat()` method
- [x] `take()` method
- [x] `put()` method
- [x] `fill()` method - fill with scalar

### Tier 3 - Conversion
- [x] `tobytes()` method - raw bytes
- [x] `view()` method - view with different dtype

### Tier 4 - Sorting
- [x] `partition()` method
- [x] `argpartition()` method

**Files**: `src/python/pyarray.rs`, `tests/test_array_methods.py`

---

## Parallel Execution with Git Worktrees

Each agent gets its own worktree so they can build/test independently.

### Setup Worktrees

```bash
# From main repo, create worktrees for each stream
git worktree add ../rumpy-stream-01 -b numpy-parity/stream-01-unary-math
git worktree add ../rumpy-stream-02 -b numpy-parity/stream-02-binary-math
git worktree add ../rumpy-stream-03 -b numpy-parity/stream-03-comparisons
# ... etc for each stream you want to run in parallel
```

Each worktree is a full working directory with isolated builds:
```
~/workspace/
  rumpy/              # main repo (coordinator)
  rumpy-stream-01/    # agent 1's worktree
  rumpy-stream-02/    # agent 2's worktree
  rumpy-stream-03/    # agent 3's worktree
```

### Launch Agents

Each agent runs in its own worktree directory:
```bash
# Terminal 1
cd ~/workspace/rumpy-stream-01
mkdir .claude
cp ../rumpy/.claude/settings.local.json .claude/
claude "Implement Stream 1 per plans/numpy-api-parity.md"

# Terminal 2
cd ~/workspace/rumpy-stream-02
cp ../rumpy/.claude/settings.local.json .claude/
claude "Implement Stream 2 per plans/numpy-api-parity.md"
```

### Merge and Cleanup

```bash
# From main repo after agents complete
cd ~/workspace/rumpy
git merge numpy-parity/stream-01-unary-math
git merge numpy-parity/stream-02-binary-math

# Cleanup worktrees when done
git worktree remove ../rumpy-stream-01
git worktree remove ../rumpy-stream-02
```

### Practical Limits

- 3-5 parallel agents recommended (disk/CPU constraints)
- Each worktree + venv + target is ~1-2GB

---

## Git Workflow

### Branch Strategy
Each stream works on its own branch:
```
numpy-parity/stream-01-unary-math
numpy-parity/stream-02-binary-math
numpy-parity/stream-03-comparisons
...
```

### Merge Order
Streams with no dependencies can merge independently. Suggested order:
1. Streams 1-2 (math ops) - foundational
2. Streams 3-5 (comparison, logic, bitwise) - similar patterns
3. Streams 6-7 (reductions, statistics) - extend existing
4. Streams 8-11 (creation, shape, indexing, manipulation)
5. Streams 12-18 (specialized)

### Conflict Resolution
Most streams touch different files. Potential conflicts:
- `src/python/mod.rs` - all streams add exports (append-only, easy merge)
- `src/array/dtype/mod.rs` - streams 1-5 add enums (coordinate placement)

---

## Agent Prompts

### Standard Agent Prompt Template
```
You are implementing NumPy API parity for rumpy, a NumPy clone in Rust.

**Your Stream**: [STREAM_NUMBER] - [STREAM_NAME]
**Branch**: numpy-parity/stream-[NUMBER]-[name]

**Reference Files**:
- `designs/adding-operations.md` - how to add operations
- `designs/ufuncs.md` - architecture overview
- `designs/iteration-performance.md` - iteration patterns
- `designs/backstride-iteration.md` - strided access
- `plans/numpy-api-parity.md` - full plan (your section)

**Philosophy**: Seek general solutions. There are many operations and dtypes - prefer
well-factored code with shared abstractions (macros, traits) over copy-paste.
Try to acheive performance parity with numpy.  When writing iterative code this
often means writing things so that they can take advantage of simd
optimizations, or better yet, using existing iteration systems in the project
that already do this.


**Your Tasks**:
[LIST OF FUNCTIONS FROM PLAN]

**Workflow**:
1. For each function:
   a. Write test in tests/ comparing against numpy
   b. Implement in Rust
   c. Add Python binding
   d. Run pytest to verify
   e. Commit when test passes
2. Run full test suite before finishing
3. When done, review your work to see if there is anything you should simplify or clean up
4. Do NOT merge - leave for review

**Testing**: Use pytest in the tests/ directory. Compare rumpy results against numpy
using `assert_eq` from `tests/helpers.py`.

When you're done, review your code to see if there is anything you should clean
up or simplify.

**Build Command**:
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
```

**Important**:
- Do NOT modify files outside your stream's scope
- If you need a dependency from another stream, note it and skip
- Commit frequently with clear messages

---

## Estimated Scope

| Stream | Functions | Complexity | Dependencies | Status |
|--------|-----------|------------|--------------|--------|
| 1. Unary Math | 17 | Low | None | ✅ |
| 2. Binary Math | 11 | Low | None | ✅ |
| 3. Comparisons | 9 | Medium | None | ✅ |
| 4. Logical | 4 | Low | None | ✅ |
| 5. Bitwise | 6 | Low | None | ✅ |
| 6. NaN Reductions | 9 | Medium | None | ✅ |
| 7. Statistics | 6 | High | Stream 6 partial | ✅ |
| 8. Creation | 11 | Medium | None | ✅ |
| 9. Shape | 11 | Medium | None | ✅ |
| 10. Indexing | 9 | High | None | ✅ |
| 11. Manipulation | 14 | Medium | None | ✅ |
| 12. Set Ops | 6 | Medium | Stream 13 | ✅ |
| 13. Sorting Adv | 3 | High | None | ✅ |
| 14. Linalg | 11 | High | None | ✅ |
| 15. Random | 8 | Medium | None | ✅ |
| 16. Numerical | 5 | High | None | ✅ |
| 17. Polynomial | 5 | High | None | ✅ |
| 18. Array Methods | 12 | Medium | Various | ✅ |
| 19. I/O | 10 | High | None | ✅ |
| 20. FFT Extensions | 8 | High | Stream fft | ✅ |
| 21. Random Extended | 18 | Medium | Stream 15 | ✅ |
| 22. DType System | 9 | High | None | ✅ |
| 23. ndarray Methods | 11 | Medium | None | ✅ |
| 24. Linalg Extensions | 9 | High | Stream 14 | ✅ |
| 25. Special Functions | 9 | Medium | None | ✅ |
| 26. Index Utilities | 10 | Medium | None | ✅ |
| 27. Array Inspection | 12 | Low | None | ✅ |
| 28. Window Functions | 5 | Low | None | ✅ |
| 29. Unique Extensions | 4 | Low | Stream 13 | ✅ |
| 30. Convenience Aliases | 15 | Low | Various | ✅ |
| 31. NaN Extensions | 5 | Medium | Stream 6 | |
| 32. Miscellaneous | 13 | Medium | Various | |

**Completed**: ~248 functions across 30 streams
**Remaining**: ~45 functions across 2 streams
**Total**: ~293 functions across 32 streams

---

## Quick Start for Single Agent

If working alone, prioritize by user impact:

**Week 1**: Streams 1-2 (math), Stream 3 (comparisons)
**Week 2**: Streams 4-5 (logic, bitwise), Stream 6 (nan reductions)
**Week 3**: Stream 9 (shape), Stream 11 (manipulation)
**Week 4**: Stream 8 (creation), Stream 10 (indexing)
**Later**: Streams 7, 12-18 (specialized)

---

## Validation Checklist

Before marking stream complete:
- [x] All functions implemented
- [x] Tests compare against NumPy
- [x] Tests cover multiple dtypes
- [x] Tests cover edge cases (empty, NaN, broadcasting)
- [x] `cargo clippy` passes
- [x] `pytest tests/ -v` passes (2400 tests)
- [x] No conflicts with main branch

---

## Stream 19: I/O Operations ✅ COMPLETE

File reading/writing - essential for real-world use.

### Tier 1 - Text I/O
- [x] `loadtxt` - load from text file
- [x] `savetxt` - save to text file
- [x] `genfromtxt` - load with missing value handling

### Tier 2 - Binary I/O
- [x] `save` - save single array (.npy format)
- [x] `load` - load .npy file
- [x] `savez` - save multiple arrays (.npz)
- [x] `savez_compressed` - compressed .npz

### Tier 3 - Buffer Operations
- [x] `frombuffer` - create from buffer
- [x] `fromfile` - read from binary file
- [x] `tofile` - array method, write binary

**Files**: `src/ops/io.rs`, `src/python/io.rs`, `tests/test_io.py`

---

## Stream 20: FFT Extensions ✅ COMPLETE

Complete the FFT submodule.

### N-dimensional
- [x] `fftn` - n-dimensional FFT
- [x] `ifftn` - inverse n-dimensional FFT
- [x] `rfftn` - real n-dimensional FFT
- [x] `irfftn` - inverse real n-dimensional FFT

### Real 2D
- [x] `rfft2` - 2D real FFT
- [x] `irfft2` - inverse 2D real FFT

### Hermitian
- [x] `hfft` - Hermitian FFT
- [x] `ihfft` - inverse Hermitian FFT

**Performance Notes**:
- 1D/2D complex FFT: competitive with NumPy (0.4x-1.5x)
- N-dimensional FFT: often faster than NumPy for small arrays
- Real FFT variants: 1.5x-5x slower for large arrays (uses full complex FFT internally)

**Files**: `src/ops/fft.rs`, `src/python/fft.rs`, `tests/test_fft.py`

---

## Stream 21: Random Distributions (Extended) ✅ COMPLETE

Additional distributions for Generator.

### Tier 1 - Common
- [x] `lognormal` - log-normal distribution
- [x] `laplace` - Laplace distribution
- [x] `logistic` - logistic distribution
- [x] `rayleigh` - Rayleigh distribution
- [x] `weibull` - Weibull distribution

### Tier 2 - Discrete
- [x] `geometric` - geometric distribution
- [x] `negative_binomial` - negative binomial
- [x] `hypergeometric` - hypergeometric distribution
- [x] `multinomial` - multinomial distribution
- [x] `zipf` - Zipf distribution

### Tier 3 - Specialized
- [x] `triangular` - triangular distribution
- [x] `vonmises` - von Mises (circular) distribution
- [x] `pareto` - Pareto distribution
- [x] `wald` - Wald (inverse Gaussian) distribution
- [x] `dirichlet` - Dirichlet distribution
- [x] `standard_t` - Student's t distribution
- [x] `standard_cauchy` - Cauchy distribution
- [x] `standard_gamma` - standard gamma (shape only)

**Performance Notes** (release mode with rand_distr/Ziggurat):
- Uniform/random: 0.79x (faster than NumPy)
- Poisson: 0.59x (faster than NumPy)
- Binomial: 0.42x (faster than NumPy)
- Normal/exponential/gamma: 1.1-1.4x (competitive)
- Complex distributions (vonmises, hypergeometric): 2-10x slower

**Files**: `src/random/*.rs`, `src/python/random.rs`, `tests/test_random.py`

---

## Stream 22: DType System Extensions ✅ COMPLETE

Type introspection and promotion functions.

### Type Info
- [x] `finfo` - floating point type info
- [x] `iinfo` - integer type info

### Type Promotion
- [x] `promote_types` - find common type
- [x] `result_type` - determine result type for operands
- [x] `can_cast` - check if cast is allowed
- [x] `common_type` - find common type for sequences

### Type Predicates
- [x] `issubdtype` - check dtype inheritance
- [x] `isdtype` - check if dtype matches kind (NumPy 2.0+)

**Performance Notes**: These are metadata functions with sub-microsecond latency.
PyO3 crossing overhead makes them 2-10x slower than NumPy C functions,
but this is negligible since they're called rarely (not in hot loops).

**Files**: `src/python/dtype.rs`, `tests/test_dtypes.py`

---

## Stream 23: ndarray Methods (Extended) ✅ COMPLETE

Complete array method parity.

### Properties
- [x] `base` - underlying array for views
- [x] `data` - buffer pointer (memoryview)
- [x] `flags` - array flags object (C_CONTIGUOUS, etc.)
- [x] `flat` - 1D iterator over elements

### Methods
- [x] `ptp` - peak-to-peak (max - min)
- [x] `compress` - select using boolean mask
- [x] `choose` - construct from index choices
- [x] `resize` - resize array in-place

### Serialization
- [x] `dump` - pickle to file
- [x] `dumps` - pickle to string
- [x] `tofile` - write to binary file (already existed)

**Performance Notes** (release mode):
- Properties: ~microsecond overhead from PyO3, acceptable for metadata operations
- `ptp`: 0.9x-2x of NumPy depending on axis
- `compress`, `choose`: faster than NumPy (0.4x)
- Small operations like `resize` have fixed overhead that dominates at small sizes

**Files**: `src/python/pyarray/*.rs`, `tests/test_array_methods.py`

---

## Stream 24: Linalg Extensions ✅ COMPLETE

Complete linear algebra submodule.

### Eigenvalues
- [x] `eigvalsh` - eigenvalues of symmetric matrix

### Matrix Operations
- [x] `matrix_power` - matrix to integer power
- [x] `multi_dot` - efficient multi-matrix dot

### Tensor Operations
- [x] `tensorinv` - tensor inverse
- [x] `tensorsolve` - tensor equation solve

### Norms (Extended)
- [x] `vector_norm` - vector norm with ord parameter
- [x] `matrix_norm` - matrix norm with ord parameter

### Singular Values
- [x] `svdvals` - singular values only

### Misc
- [x] `LinAlgError` - exception class

**Files**: `src/ops/linalg.rs`, `src/python/linalg.rs`, `tests/test_linalg.py`

**Performance Notes**:
- `vector_norm`: Faster than NumPy (0.4-0.5x)
- `matrix_norm` 1/inf: Faster than NumPy (0.3-0.7x)
- `multi_dot`, `svdvals`: Competitive with NumPy
- `eigvalsh`: 2x slower (faer library overhead)
- All functions work for float32/float64, with fast paths for contiguous float64

---

## Stream 25: Special Functions ✅ COMPLETE

Mathematical special functions.

### Tier 1 - Common
- [x] `sinc` - sinc function (sin(πx)/(πx))
- [x] `i0` - modified Bessel function of order 0

### Tier 2 - Integer Math
- [x] `gcd` - greatest common divisor
- [x] `lcm` - least common multiple

### Tier 3 - Float Decomposition
- [x] `modf` - fractional and integer parts
- [x] `frexp` - mantissa and exponent
- [x] `ldexp` - x * 2^i

### Tier 4 - Special Values
- [x] `heaviside` - Heaviside step function
- [x] `spacing` - ULP distance

**Performance Notes**:
- `i0` (Bessel): Faster than NumPy (0.5-0.85x) using Chebyshev approximation
- Other functions: 6-40x slower due to `to_vec()` copy overhead
- Acceptable for low-frequency special math functions

**Files**: `src/ops/special.rs`, `src/python/ufuncs.rs`, `tests/test_special.py`

---

## Stream 26: Index Utilities ✅ COMPLETE

Advanced indexing helpers.

### Index Conversion
- [x] `unravel_index` - flat index to multi-index
- [x] `ravel_multi_index` - multi-index to flat index

### Index Generation
- [x] `diag_indices` - diagonal indices
- [x] `diag_indices_from` - diagonal indices matching array
- [x] `tril_indices` - lower triangle indices
- [x] `triu_indices` - upper triangle indices
- [x] `tril_indices_from` - lower triangle indices from array
- [x] `triu_indices_from` - upper triangle indices from array
- [x] `mask_indices` - indices from mask function

### Binning
- [x] `digitize` - bin indices for values

### Bit Packing
- [x] `packbits` - pack binary values
- [x] `unpackbits` - unpack to binary array

**Performance Notes**: These are index utility functions used for setup, not hot loops.
Performance is acceptable for typical use (5-350us per call).

**Files**: `src/ops/indexing.rs`, `src/python/indexing.rs`, `tests/test_index_utils.py`

---

## Stream 27: Array Inspection ✅ COMPLETE

Value checking and inspection functions.

### Special Value Checks
- [x] `isneginf` - check for negative infinity
- [x] `isposinf` - check for positive infinity
- [x] `isreal` - check if real (imag == 0)
- [x] `iscomplex` - check if has imaginary part
- [x] `isrealobj` - check if array is real-valued
- [x] `iscomplexobj` - check if array is complex-valued

### Memory Inspection
- [x] `shares_memory` - check if arrays share memory
- [x] `may_share_memory` - check if arrays might share memory

### Scalar/Shape Checks
- [x] `isscalar` - check if scalar
- [x] `ndim` - number of dimensions (module-level)
- [x] `size` - total elements (module-level)
- [x] `shape` - shape tuple (module-level)

**Files**: `src/python/ufuncs.rs`, `src/python/mod.rs`, `tests/test_inspection.py`

---

## Stream 28: Window Functions ✅ COMPLETE

Signal processing window functions.

- [x] `bartlett` - Bartlett window
- [x] `blackman` - Blackman window
- [x] `hamming` - Hamming window
- [x] `hanning` - Hann window
- [x] `kaiser` - Kaiser window (requires beta parameter)

**Performance Notes**:
- Small windows (100 pts): 0.12x-0.28x (much faster than NumPy)
- Large windows (10000 pts): 0.16x-1.04x (faster or competitive)
- Kaiser especially fast due to direct Bessel computation

**Files**: `src/python/creation.rs`, `src/python/mod.rs`, `tests/test_windows.py`

---

## Stream 29: Unique Extensions ✅ COMPLETE

Extended unique functionality (NumPy 2.0+).

- [x] `unique_all` - unique with all return values
- [x] `unique_counts` - unique with counts only
- [x] `unique_inverse` - unique with inverse only
- [x] `unique_values` - unique values only

**Performance Notes**: All functions return sorted unique values with O(n log n) complexity.
Returns namedtuples for counts/inverse/all variants. inverse_indices preserves input shape.

**Files**: `src/ops/array_methods/sorting.rs`, `src/python/shape.rs`, `tests/test_sorting.py`

---

## Stream 30: Convenience Aliases ✅ COMPLETE

Aliases and simple wrappers for compatibility.

### Math Aliases
- [x] `absolute` - alias for abs
- [x] `conjugate` - alias for conj
- [x] `acos`, `asin`, `atan` - aliases for arccos, arcsin, arctan
- [x] `acosh`, `asinh`, `atanh` - aliases for arccosh, arcsinh, arctanh
- [x] `pow` - alias for power
- [x] `mod` - alias for remainder
- [x] `fabs` - absolute value (float only)
- [x] `true_divide` - alias for divide
- [x] `fmod` - floating point remainder

### Reduction Aliases
- [x] `amax` - alias for max (with axis)
- [x] `amin` - alias for min (with axis)

### Constants
- [x] `pi` - π
- [x] `e` - Euler's number
- [x] `inf` - positive infinity
- [x] `nan` - Not a Number
- [x] `newaxis` - None (already existed)

**Files**: `src/python/mod.rs`, `src/python/ufuncs.rs`, `src/python/reductions.rs`, `tests/test_aliases.py`

---

## Stream 31: NaN-aware Extensions ✅ COMPLETE

Additional NaN-handling functions.

- [x] `nanmedian` - median ignoring NaN
- [x] `nanpercentile` - percentile ignoring NaN
- [x] `nanquantile` - quantile ignoring NaN
- [x] `nancumsum` - cumulative sum ignoring NaN
- [x] `nancumprod` - cumulative product ignoring NaN

**Files**: `src/ops/statistics.rs`, `src/ops/array_methods/cumulative.rs`, `src/array/manipulation.rs`, `src/python/reductions.rs`, `tests/test_nan_extensions.py`

---

## Stream 32: Miscellaneous Operations ✅ COMPLETE

Remaining useful functions.

### Array Manipulation
- [x] `resize` - resize array (different from reshape)
- [x] `unstack` - unstack along axis
- [x] `block` - assemble from nested blocks
- [x] `trim_zeros` - trim leading/trailing zeros

### Conditional
- [x] `extract` - extract elements where condition
- [x] `place` - place values where condition
- [x] `putmask` - put values using mask
- [x] `select` - select from choicelist by conditions
- [x] `piecewise` - piecewise function

### Other
- [x] `ediff1d` - differences with prepend/append
- [x] `unwrap` - unwrap phase angles
- [x] `angle` - phase angle of complex
- [x] `real_if_close` - convert to real if imaginary is small

**Performance Notes**:
- `resize`, `ediff1d` use typed dispatch for ~1.1x numpy performance
- `extract`, `place`, `putmask` use typed dispatch for contiguous arrays
- `trim_zeros`, `unwrap`, `angle` are faster than numpy (0.2x-0.5x)

**Files**: `src/python/misc.rs`, `tests/test_misc.py`

---

## Stream Summary (New)

| Stream | Functions | Focus |
|--------|-----------|-------|
| 19. I/O | 10 | File operations |
| 20. FFT Extensions | 8 | Complete FFT module |
| 21. Random Extended | 18 | More distributions |
| 22. DType System | 9 | Type introspection |
| 23. ndarray Methods | 11 | Array method parity |
| 24. Linalg Extensions | 9 | Complete linalg |
| 25. Special Functions | 9 | Math special functions |
| 26. Index Utilities | 10 | Advanced indexing |
| 27. Array Inspection | 12 | Value/memory checks |
| 28. Window Functions | 5 | Signal windows |
| 29. Unique Extensions | 4 | NumPy 2.0 unique |
| 30. Convenience Aliases | 15 | Compatibility |
| 31. NaN Extensions | 5 | More NaN-aware ops |
| 32. Miscellaneous | 13 | Remaining useful |

**New Total**: ~128 additional functions across 14 new streams

---

## Code Quality Issues

Post-implementation cleanup opportunities identified during review.

### 1. Large Files ✅ FIXED

Broken these files apart into different files based on grouped functionality.

| File | Lines | Issue |
|------|-------|-------|
| `src/python/mod.rs` | 3433 | All Python bindings in one file |
| `src/ops/mod.rs` | 2405 | Many array methods mixed with module setup |
| `src/array/mod.rs` | 2321 | Core array + creation + manipulation |

**Potential refactoring:**
- [ ] Split `src/python/mod.rs` by category (creation, math, shape, etc.)
- [ ] Extract array methods from `src/ops/mod.rs` into focused submodules
- [ ] Consider `src/array/creation.rs` for array factory functions

**Trade-off:** More files vs. easier navigation. Current structure works but may become unwieldy.

### 2. Duplicate Test Files ✅ FIXED

Consolidated `test_stats.py` into `test_statistics.py`.

### 3. Open TODOs in Code

| Location | TODO | Status |
|----------|------|--------|
| `src/array/mod.rs:1055` | dtype promotion for concatenate | Open |
| `src/python/pyarray.rs:1603` | scalar op elision | Open (optimization) |
| `src/python/linalg.rs:30` | full_matrices support in SVD | Open |

**Priority:**
- [ ] dtype promotion - affects correctness when concatenating mixed types
- [ ] full_matrices SVD - NumPy parity gap
- [x] ~~dtype conversion in asarray~~ - Fixed

### 4. Deprecation Warnings ✅ FIXED

Added deprecation warnings for:
- `row_stack` → use `vstack`
- `in1d` → use `isin`

### 5. Integer dtype behavior ✅ FIXED

`deg2rad`/`radians`/`rad2deg`/`degrees` now promote integer inputs to float64,
matching NumPy behavior.
