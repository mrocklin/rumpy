"""Testing utilities for rumpy - compare against numpy.

See designs/testing.md for testing philosophy.
"""

import numpy as np
import rumpy as rp


def assert_eq(a, b, rtol=1e-7, atol=1e-14):
    """
    Assert two array-likes are equal, converting rumpy arrays to numpy.

    Parameters
    ----------
    a, b : array-like
        Arrays to compare. Can be rumpy.ndarray, numpy.ndarray, or scalar.
    rtol : float
        Relative tolerance for floating point comparison.
    atol : float
        Absolute tolerance for floating point comparison.
    """
    a_np = np.asarray(a) if not isinstance(a, np.ndarray) else a
    b_np = np.asarray(b) if not isinstance(b, np.ndarray) else b

    assert a_np.shape == b_np.shape, f"Shape mismatch: {a_np.shape} vs {b_np.shape}"

    if np.issubdtype(a_np.dtype, np.floating) or np.issubdtype(
        a_np.dtype, np.complexfloating
    ):
        np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol, equal_nan=True)
    else:
        np.testing.assert_array_equal(a_np, b_np)


def assert_dtype_eq(r, n):
    """Assert rumpy and numpy arrays have matching dtypes."""
    r_dtype = r.dtype if hasattr(r, "dtype") else type(r).__name__
    n_dtype = str(n.dtype) if hasattr(n, "dtype") else type(n).__name__
    assert str(r_dtype) == str(n_dtype), f"Dtype mismatch: {r_dtype} vs {n_dtype}"


def make_numpy(shape, dtype):
    """
    Generate a numpy array with non-trivial values.

    Values are chosen to be safe for the dtype and avoid edge cases
    like division by zero. Good for general-purpose testing.
    """
    size = int(np.prod(shape)) if shape else 1

    if dtype == "bool":
        data = np.array([i % 2 == 0 for i in range(size)], dtype=bool)
    elif "complex" in dtype:
        data = np.array([complex(i + 1, i + 2) for i in range(size)], dtype=dtype)
    elif "uint" in dtype:
        data = np.array([i + 1 for i in range(size)], dtype=dtype)
    elif "int" in dtype:
        data = np.array([i - size // 2 for i in range(size)], dtype=dtype)
    else:  # float
        data = np.array([(i + 1) * 0.5 for i in range(size)], dtype=dtype)

    return data.reshape(shape) if shape else data


def make_pair(shape, dtype):
    """
    Create matching (rumpy, numpy) array pair with non-trivial values.

    Returns
    -------
    r : rumpy.ndarray
    n : numpy.ndarray
    """
    n = make_numpy(shape, dtype)
    r = rp.asarray(n)
    return r, n


def make_positive(shape, dtype):
    """Generate array with strictly positive values (safe for sqrt, log, etc.)."""
    size = int(np.prod(shape)) if shape else 1

    if dtype == "bool":
        data = np.ones(size, dtype=bool)
    elif "complex" in dtype:
        data = np.array([complex(i + 1, i + 1) for i in range(size)], dtype=dtype)
    elif "int" in dtype or "uint" in dtype:
        data = np.array([i + 1 for i in range(size)], dtype=dtype)
    else:
        data = np.array([(i + 1) * 0.5 for i in range(size)], dtype=dtype)

    return data.reshape(shape) if shape else data


def make_positive_pair(shape, dtype):
    """Create matching pair with strictly positive values."""
    n = make_positive(shape, dtype)
    r = rp.asarray(n)
    return r, n
