"""Tests for dot product with numpy semantics."""

import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestDot:
    """Tests for dot()."""

    def test_1d_1d(self):
        """Inner product of two 1D arrays."""
        a = rp.asarray([1.0, 2.0, 3.0])
        b = rp.asarray([4.0, 5.0, 6.0])
        r = rp.dot(a, b)

        na = np.array([1.0, 2.0, 3.0])
        nb = np.array([4.0, 5.0, 6.0])
        n = np.dot(na, nb)

        # Result should be scalar (0D array)
        assert r.ndim == 0
        assert abs(float(r) - n) < 1e-10

    def test_2d_2d(self):
        """Matrix multiplication via dot."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        B = rp.asarray([[5.0, 6.0], [7.0, 8.0]])
        r = rp.dot(A, B)

        nA = np.array([[1.0, 2.0], [3.0, 4.0]])
        nB = np.array([[5.0, 6.0], [7.0, 8.0]])
        n = np.dot(nA, nB)
        assert_eq(r, n)

    def test_1d_2d(self):
        """Vector-matrix product."""
        a = rp.asarray([1.0, 2.0])
        B = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        r = rp.dot(a, B)

        na = np.array([1.0, 2.0])
        nB = np.array([[1.0, 2.0], [3.0, 4.0]])
        n = np.dot(na, nB)
        assert_eq(r, n)

    def test_2d_1d(self):
        """Matrix-vector product."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        b = rp.asarray([1.0, 2.0])
        r = rp.dot(A, b)

        nA = np.array([[1.0, 2.0], [3.0, 4.0]])
        nb = np.array([1.0, 2.0])
        n = np.dot(nA, nb)
        assert_eq(r, n)

    def test_matches_matmul_2d(self):
        """For 2D arrays, dot should match matmul."""
        A = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        B = rp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        r_dot = rp.dot(A, B)
        r_matmul = rp.matmul(A, B)
        assert_eq(r_dot, r_matmul)
