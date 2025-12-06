# Deviations from NumPy

Intentional differences from NumPy behavior.

## `inner` - Gufunc Broadcasting vs Cartesian Product

NumPy's `inner(a, b)` for multi-dimensional arrays computes inner products over all combinations of batch dimensions (Cartesian product). For example:

```python
np.inner(shape(2,3), shape(4,3)) -> shape(2,4)
```

Rumpy's `inner` uses standard gufunc broadcasting instead:

```python
rp.inner(shape(2,3), shape(4,3)) -> error (incompatible loop dims)
rp.inner(shape(2,3), shape(2,3)) -> shape(2,)  # broadcasts
rp.inner(shape(2,3), shape(3,))  -> shape(2,)  # broadcasts
```

This is simpler and consistent with how gufuncs work. For the Cartesian product behavior, use explicit loops or `tensordot`.

## dtype Strings Only

Rumpy accepts dtype as strings (`"float64"`) not NumPy dtype objects (`np.float64`).

## Random Number Generation

`random()` and `uniform()` match numpy exactly when using `Generator.from_numpy_state()` with state extracted from numpy's PCG64DXSM.

`integers()` uses Lemire's algorithm for bounded integers, which differs from numpy's algorithm. Values are uniformly distributed but not bit-identical.

`normal()` and `exponential()` use Box-Muller and inverse transform respectively, rather than numpy's ziggurat algorithm. Statistically equivalent but not bit-identical.

## `round` - Half-Rounding Behavior

NumPy uses "round half to even" (banker's rounding) where 0.5 rounds to the nearest even number:
```python
np.round([1.5, 2.5, 3.5, 4.5])  # -> [2., 2., 4., 4.]
```

Rust (and rumpy) uses "round half away from zero":
```python
rp.round([1.5, 2.5, 3.5, 4.5])  # -> [2., 3., 4., 5.]
```

This only affects values exactly at 0.5. For most use cases the difference is negligible.

## Temporary Array Elision - No Stack Unwinding

NumPy's temporary elision (reusing buffers for intermediates like `x + 1 + 2`) uses expensive stack unwinding (~35μs) to verify no C extension holds a raw pointer to the buffer. This makes the optimization safe even when C code retains pointers without incrementing refcounts.

Rumpy skips stack unwinding and relies solely on Python refcount checking. This means:

1. **Faster check** - No stack unwinding overhead
2. **Same size threshold** - Only applies to arrays ≥256KB
3. **Theoretical unsafety** - If a C extension holds a raw pointer without incrementing refcount, the buffer could be overwritten

In practice this is safe because:
- Well-behaved C code increments refcounts when holding references
- Most extensions use the buffer protocol properly
- Chained pure-Python ops (the target use case) don't involve C mid-chain

For safety-critical code with C extensions that might hold raw pointers, assign intermediates to variables to prevent elision.
