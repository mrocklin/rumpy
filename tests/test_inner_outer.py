"""Tests for inner and outer product gufuncs."""

import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestInner:
    """Tests for inner product."""

    def test_1d_1d(self):
        """Basic 1D inner product (dot product)."""
        r = rp.inner(rp.asarray([1, 2, 3]), rp.asarray([4, 5, 6]))
        n = np.inner(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert_eq(r, n)

    def test_1d_values(self):
        """Check actual values."""
        a = rp.asarray([1.0, 2.0, 3.0])
        b = rp.asarray([4.0, 5.0, 6.0])
        result = rp.inner(a, b)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        n = np.inner(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        assert_eq(result, n)

    def test_2d_1d(self):
        """2D array inner with 1D array."""
        a = rp.arange(6, dtype="float64").reshape(2, 3)
        b = rp.asarray([1.0, 2.0, 3.0])
        r = rp.inner(a, b)
        n = np.inner(np.arange(6, dtype="float64").reshape(2, 3), np.array([1.0, 2.0, 3.0]))
        assert_eq(r, n)

    def test_batched(self):
        """Batched inner product with compatible loop dims."""
        # (3, 4) inner (3, 4) -> (3,) - broadcasts loop dim
        a = rp.arange(12, dtype="float64").reshape(3, 4)
        b = rp.arange(12, dtype="float64").reshape(3, 4) + 1
        r = rp.inner(a, b)
        # Compute expected: inner product along last axis for each row
        na = np.arange(12, dtype="float64").reshape(3, 4)
        nb = np.arange(12, dtype="float64").reshape(3, 4) + 1
        # Manual: for each row i, sum a[i,:] * b[i,:]
        expected = np.array([np.dot(na[i], nb[i]) for i in range(3)])
        assert_eq(r, expected)


class TestOuter:
    """Tests for outer product."""

    def test_1d_1d(self):
        """Basic 1D outer product."""
        r = rp.outer(rp.asarray([1, 2, 3]), rp.asarray([4, 5]))
        n = np.outer(np.array([1, 2, 3]), np.array([4, 5]))
        assert_eq(r, n)

    def test_values(self):
        """Check actual values."""
        a = rp.asarray([1.0, 2.0, 3.0])
        b = rp.asarray([4.0, 5.0])
        result = rp.outer(a, b)
        expected = np.outer(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0]))
        # [[4, 5], [8, 10], [12, 15]]
        assert_eq(result, expected)

    def test_flattens_2d(self):
        """Outer flattens higher-D arrays (like numpy)."""
        a = rp.arange(4, dtype="float64").reshape(2, 2)
        b = rp.arange(3, dtype="float64")
        r = rp.outer(a, b)
        n = np.outer(np.arange(4, dtype="float64").reshape(2, 2),
                     np.arange(3, dtype="float64"))
        assert_eq(r, n)

    def test_single_element(self):
        """Outer product with single elements."""
        r = rp.outer(rp.asarray([5.0]), rp.asarray([3.0]))
        n = np.outer(np.array([5.0]), np.array([3.0]))
        assert_eq(r, n)
