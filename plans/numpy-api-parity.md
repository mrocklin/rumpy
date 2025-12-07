# NumPy API Parity Plan

**Design Ref**: `designs/adding-operations.md`, `designs/ufuncs.md`

## Overview

This plan covers ~130 missing NumPy functions organized into parallelizable work streams.
Each stream is independent and can be worked on by a separate agent.

---

## Stream 1: Math Operations (Unary)

Simple element-wise operations using `map_unary`. Pattern: add to `UnaryOp` enum,
implement in dtype files, add Python binding, test.

### Tier 1 - Common
- [ ] `square` - x²
- [ ] `negative` - -x
- [ ] `positive` - +x (identity for numeric)
- [ ] `reciprocal` - 1/x
- [ ] `abs`/`absolute` - already have abs, add alias

### Tier 2 - Precision Variants
- [ ] `exp2` - 2^x
- [ ] `expm1` - e^x - 1 (precise for small x)
- [ ] `log1p` - log(1 + x) (precise for small x)
- [ ] `cbrt` - cube root

### Tier 3 - Rounding
- [ ] `trunc` - truncate toward zero
- [ ] `rint` - round to nearest integer
- [ ] `fix` - round toward zero (alias for trunc)

### Tier 4 - Inverse Hyperbolic
- [ ] `arcsinh` - inverse sinh
- [ ] `arccosh` - inverse cosh
- [ ] `arctanh` - inverse tanh

### Tier 5 - Misc
- [ ] `signbit` - true if sign bit set
- [ ] `nan_to_num` - replace nan/inf with numbers

**Files**: `src/array/dtype/mod.rs`, `src/array/dtype/*.rs`, `src/ops/registry.rs`,
`src/python/pyarray.rs`, `tests/test_math.py`

---

## Stream 2: Math Operations (Binary)

Element-wise binary ops using `map_binary` with broadcasting.

### Tier 1 - Essential
- [ ] `arctan2` - two-argument arctangent
- [ ] `hypot` - sqrt(x² + y²)
- [ ] `mod` - alias for remainder
- [ ] `fmax` - element-wise max ignoring NaN
- [ ] `fmin` - element-wise min ignoring NaN
- [ ] `copysign` - copy sign from y to x

### Tier 2 - Precision
- [ ] `logaddexp` - log(exp(x) + exp(y))
- [ ] `logaddexp2` - log2(2^x + 2^y)
- [ ] `nextafter` - next floating point value toward y

### Tier 3 - Angular
- [ ] `deg2rad` / `radians` - degrees to radians
- [ ] `rad2deg` / `degrees` - radians to degrees

**Files**: Same as Stream 1 plus `src/ops/mod.rs` for mixed-type fallbacks.

---

## Stream 3: Comparison Operations

Return boolean arrays. Use `map_binary` with bool output.

### Core Comparisons
- [ ] `equal` - x == y (element-wise)
- [ ] `not_equal` - x != y
- [ ] `less` - x < y
- [ ] `less_equal` - x <= y
- [ ] `greater` - x > y
- [ ] `greater_equal` - x >= y

### Approximate Comparisons
- [ ] `isclose` - element-wise approximate equality
- [ ] `allclose` - all elements approximately equal (returns scalar)
- [ ] `array_equal` - arrays have same shape and elements

**Note**: The `__lt__`, `__eq__` etc. operators may already exist. These are the
function forms that can take `out` parameter and work with `where`.

**Files**: `src/array/dtype/mod.rs` (add ComparisonOp enum), `src/ops/comparison.rs` (new),
`src/python/mod.rs`, `tests/test_comparison.py`

---

## Stream 4: Logical Operations

Boolean logic, both element-wise and reduction.

### Element-wise
- [ ] `logical_and` - element-wise AND
- [ ] `logical_or` - element-wise OR
- [ ] `logical_not` - element-wise NOT
- [ ] `logical_xor` - element-wise XOR

**Files**: `src/ops/logical.rs` (new), `src/python/mod.rs`

---

## Stream 5: Bitwise Operations

Integer-only bit manipulation.

- [ ] `bitwise_and` - &
- [ ] `bitwise_or` - |
- [ ] `bitwise_xor` - ^
- [ ] `bitwise_not` / `invert` - ~
- [ ] `left_shift` - <<
- [ ] `right_shift` - >>

**Files**: `src/ops/bitwise.rs` (new), `src/python/mod.rs`, `tests/test_bitwise.py`

---

## Stream 6: Reductions (NaN-aware)

Same as existing reductions but ignore NaN values.

- [ ] `nansum` - sum ignoring NaN
- [ ] `nanprod` - product ignoring NaN
- [ ] `nanmean` - mean ignoring NaN
- [ ] `nanstd` - std ignoring NaN
- [ ] `nanvar` - var ignoring NaN
- [ ] `nanmin` - min ignoring NaN
- [ ] `nanmax` - max ignoring NaN
- [ ] `nanargmin` - argmin ignoring NaN
- [ ] `nanargmax` - argmax ignoring NaN

**Pattern**: Wrap existing reduction with NaN check in accumulator.

**Files**: `src/ops/reductions.rs`, `src/python/mod.rs`, `tests/test_nan.py`

---

## Stream 7: Statistical Operations

- [ ] `median` - median value (requires sorting)
- [ ] `average` - weighted average
- [ ] `ptp` - peak-to-peak (max - min)
- [ ] `histogram` - compute histogram
- [ ] `corrcoef` - correlation coefficient matrix
- [ ] `cov` - covariance matrix

**Files**: `src/ops/statistics.rs` (new), `src/python/mod.rs`, `tests/test_statistics.py`

---

## Stream 8: Array Creation

### Tier 1 - Common
- [ ] `full_like` - like full but match shape/dtype of input
- [ ] `identity` - identity matrix
- [ ] `logspace` - logarithmically spaced values
- [ ] `geomspace` - geometrically spaced values

### Tier 2 - Triangular
- [ ] `tri` - triangular matrix of ones
- [ ] `tril` - lower triangle
- [ ] `triu` - upper triangle
- [ ] `diagflat` - create diagonal matrix from flat input

### Tier 3 - Advanced
- [ ] `meshgrid` - coordinate matrices from vectors
- [ ] `indices` - grid of indices
- [ ] `fromfunction` - construct from function

**Files**: `src/python/mod.rs`, `tests/test_creation.py`

---

## Stream 9: Shape Manipulation (Module-level)

Module-level versions of array methods + new functions.

### Tier 1 - Aliases
- [ ] `reshape` - module-level reshape
- [ ] `ravel` - flatten to 1D (return view if possible)
- [ ] `flatten` - flatten to 1D (always copy)
- [ ] `transpose` - module-level transpose

### Tier 2 - Dimension Manipulation
- [ ] `atleast_1d` - ensure at least 1D
- [ ] `atleast_2d` - ensure at least 2D
- [ ] `atleast_3d` - ensure at least 3D
- [ ] `moveaxis` - move axis to new position
- [ ] `rollaxis` - roll axis backward

### Tier 3 - Broadcasting
- [ ] `broadcast_to` - broadcast array to shape
- [ ] `broadcast_arrays` - broadcast multiple arrays together

**Files**: `src/python/mod.rs`, `tests/test_shape.py`

---

## Stream 10: Indexing Operations

### Tier 1 - Selection
- [ ] `take` - take elements along axis
- [ ] `take_along_axis` - take using index array along axis
- [ ] `compress` - select elements using boolean mask along axis

### Tier 2 - Search
- [ ] `searchsorted` - find insertion points for sorted array
- [ ] `argwhere` - indices where condition is true
- [ ] `flatnonzero` - indices of non-zero elements in flattened array

### Tier 3 - Modification
- [ ] `put` - replace elements at indices
- [ ] `put_along_axis` - put values using index array
- [ ] `choose` - construct array from index array and choices

**Files**: `src/ops/indexing.rs` (new), `src/python/mod.rs`, `tests/test_indexing.py`

---

## Stream 11: Array Manipulation

### Tier 1 - Splitting
- [ ] `hsplit` - horizontal split (along axis 1)
- [ ] `vsplit` - vertical split (along axis 0)
- [ ] `dsplit` - depth split (along axis 2)

### Tier 2 - Stacking
- [ ] `column_stack` - stack 1D as columns
- [ ] `row_stack` - alias for vstack
- [ ] `dstack` - stack along third axis

### Tier 3 - Repetition
- [ ] `repeat` - repeat elements
- [ ] `tile` - tile array

### Tier 4 - Modification
- [ ] `append` - append to array
- [ ] `insert` - insert into array
- [ ] `delete` - delete from array
- [ ] `pad` - pad array

### Tier 5 - Rotation
- [ ] `roll` - roll elements along axis
- [ ] `rot90` - rotate 90 degrees

**Files**: `src/python/mod.rs`, `tests/test_manipulation.py`

---

## Stream 12: Set Operations

- [ ] `isin` - test if elements are in test array
- [ ] `in1d` - test if elements are in 1D array (deprecated, alias to isin)
- [ ] `intersect1d` - intersection of sorted arrays
- [ ] `union1d` - union of arrays
- [ ] `setdiff1d` - set difference
- [ ] `setxor1d` - set symmetric difference

**Files**: `src/ops/set_ops.rs` (new), `src/python/mod.rs`, `tests/test_set.py`

---

## Stream 13: Sorting/Searching Advanced

- [ ] `partition` - partial sort
- [ ] `argpartition` - indices for partial sort
- [ ] `lexsort` - indirect sort using multiple keys

**Files**: `src/ops/sorting.rs`, `src/python/mod.rs`, `tests/test_sorting.py`

---

## Stream 14: Linear Algebra Extensions

### linalg submodule
- [ ] `eig` - general eigenvalue decomposition
- [ ] `eigvals` - eigenvalues only
- [ ] `lstsq` - least squares solution
- [ ] `pinv` - Moore-Penrose pseudo-inverse
- [ ] `matrix_rank` - matrix rank
- [ ] `cond` - condition number
- [ ] `slogdet` - sign and log of determinant

### Module-level
- [ ] `tensordot` - tensor dot product
- [ ] `vdot` - vector dot product (conjugate first arg)
- [ ] `kron` - Kronecker product
- [ ] `cross` - cross product

**Files**: `src/linalg/*.rs`, `src/python/linalg.rs`, `tests/test_linalg.py`

---

## Stream 15: Random Extensions

### Generator methods
- [ ] `permutation` - random permutation
- [ ] `shuffle` - shuffle in place
- [ ] `beta` - beta distribution
- [ ] `gamma` - gamma distribution
- [ ] `poisson` - Poisson distribution
- [ ] `binomial` - binomial distribution
- [ ] `chisquare` - chi-square distribution
- [ ] `multivariate_normal` - multivariate normal

**Files**: `src/random/*.rs`, `src/python/random.rs`, `tests/test_random.py`

---

## Stream 16: Numerical Operations

- [ ] `gradient` - numerical gradient
- [ ] `trapezoid` - trapezoidal integration
- [ ] `interp` - 1D linear interpolation
- [ ] `convolve` - already have, verify edge cases
- [ ] `correlate` - cross-correlation

**Files**: `src/ops/numerical.rs` (new), `src/python/mod.rs`, `tests/test_numerical.py`

---

## Stream 17: Polynomial Operations

- [ ] `polyfit` - polynomial curve fitting
- [ ] `polyval` - evaluate polynomial
- [ ] `polyder` - polynomial derivative
- [ ] `polyint` - polynomial integral
- [ ] `roots` - polynomial roots

**Files**: `src/ops/poly.rs` (new), `src/python/mod.rs`, `tests/test_poly.py`

---

## Stream 18: ndarray Methods

Methods missing from the array class itself.

### Tier 1 - Index operations
- [ ] `nonzero()` method - indices of non-zero elements
- [ ] `argsort()` method - already have? verify
- [ ] `sort()` method - in-place sort
- [ ] `searchsorted()` method

### Tier 2 - Element operations
- [ ] `repeat()` method
- [ ] `take()` method
- [ ] `put()` method
- [ ] `fill()` method - fill with scalar

### Tier 3 - Conversion
- [ ] `tobytes()` method - raw bytes
- [ ] `view()` method - view with different dtype

### Tier 4 - Sorting
- [ ] `partition()` method
- [ ] `argpartition()` method

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
