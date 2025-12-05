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
