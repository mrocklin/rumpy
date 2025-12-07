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

**Build Command**:
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop
```

**Important**:
- Do NOT modify files outside your stream's scope
- If you need a dependency from another stream, note it and skip
- Commit frequently with clear messages
```

---

## Estimated Scope

| Stream | Functions | Complexity | Dependencies |
|--------|-----------|------------|--------------|
| 1. Unary Math | 17 | Low | None |
| 2. Binary Math | 11 | Low | None |
| 3. Comparisons | 9 | Medium | None |
| 4. Logical | 4 | Low | None |
| 5. Bitwise | 6 | Low | None |
| 6. NaN Reductions | 9 | Medium | None |
| 7. Statistics | 6 | High | Stream 6 partial |
| 8. Creation | 11 | Medium | None |
| 9. Shape | 11 | Medium | None |
| 10. Indexing | 9 | High | None |
| 11. Manipulation | 14 | Medium | None |
| 12. Set Ops | 6 | Medium | Stream 13 |
| 13. Sorting Adv | 3 | High | None |
| 14. Linalg | 11 | High | None |
| 15. Random | 8 | Medium | None |
| 16. Numerical | 5 | High | None |
| 17. Polynomial | 5 | High | None |
| 18. Array Methods | 12 | Medium | Various |

**Total**: ~157 functions across 18 streams

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
- [ ] All functions implemented
- [ ] Tests compare against NumPy
- [ ] Tests cover multiple dtypes
- [ ] Tests cover edge cases (empty, NaN, broadcasting)
- [ ] `cargo clippy` passes
- [ ] `pytest tests/ -v` passes
- [ ] No conflicts with main branch
