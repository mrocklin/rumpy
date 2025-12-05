"""Testing utilities for rumpy - compare against numpy."""

import numpy as np


def assert_eq(a, b, rtol=1e-7, atol=1e-14):
    """
    Assert two array-likes are equal, converting rumpy arrays to numpy.

    Follows Dask's pattern: flexible input, clear error messages.

    Parameters
    ----------
    a, b : array-like
        Arrays to compare. Can be rumpy.ndarray, numpy.ndarray, or list.
    rtol : float
        Relative tolerance for floating point comparison.
    atol : float
        Absolute tolerance for floating point comparison.
    """
    # Convert to numpy via __array_interface__ or np.asarray
    a_np = np.asarray(a) if not isinstance(a, np.ndarray) else a
    b_np = np.asarray(b) if not isinstance(b, np.ndarray) else b

    # Check shapes first (better error message)
    assert a_np.shape == b_np.shape, (
        f"Shape mismatch: {a_np.shape} vs {b_np.shape}"
    )

    # Check dtypes
    # Allow compatible types but warn if different
    if a_np.dtype != b_np.dtype:
        # For now, just note the difference - values should still match
        pass

    # For floating point or complex, use allclose with NaN handling
    if np.issubdtype(a_np.dtype, np.floating) or np.issubdtype(a_np.dtype, np.complexfloating):
        np.testing.assert_allclose(
            a_np, b_np, rtol=rtol, atol=atol, equal_nan=True
        )
    else:
        # Exact equality for integers/bools
        np.testing.assert_array_equal(a_np, b_np)
