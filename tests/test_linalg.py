"""Tests for linear algebra functions: trace, det, norm."""

import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestTrace:
    """Tests for trace (sum of diagonal)."""

    def test_identity(self):
        """Trace of identity is n."""
        r = rp.trace(rp.eye(3))
        n = np.trace(np.eye(3))
        assert abs(r - n) < 1e-10

    def test_2x2(self):
        """Simple 2x2 matrix."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        nA = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert abs(rp.trace(A) - np.trace(nA)) < 1e-10

    def test_3x3(self):
        """3x3 matrix."""
        A = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        nA = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        assert abs(rp.trace(A) - np.trace(nA)) < 1e-10

    def test_rectangular(self):
        """Trace of rectangular matrix."""
        A = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        nA = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert abs(rp.trace(A) - np.trace(nA)) < 1e-10


class TestDet:
    """Tests for determinant."""

    def test_identity(self):
        """Determinant of identity is 1."""
        r = rp.det(rp.eye(3))
        n = np.linalg.det(np.eye(3))
        assert abs(r - n) < 1e-10

    def test_2x2(self):
        """2x2 determinant."""
        A = rp.asarray([[3.0, 1.0], [1.0, 2.0]])
        nA = np.array([[3.0, 1.0], [1.0, 2.0]])
        assert abs(rp.det(A) - np.linalg.det(nA)) < 1e-10

    def test_3x3(self):
        """3x3 determinant."""
        A = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        nA = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        assert abs(rp.det(A) - np.linalg.det(nA)) < 1e-10

    def test_singular(self):
        """Determinant of singular matrix is 0."""
        A = rp.asarray([[1.0, 2.0], [2.0, 4.0]])  # rank 1
        nA = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert abs(rp.det(A) - np.linalg.det(nA)) < 1e-10

    def test_negative_det(self):
        """Matrix with negative determinant."""
        A = rp.asarray([[0.0, 1.0], [1.0, 0.0]])  # permutation, det = -1
        nA = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert abs(rp.det(A) - np.linalg.det(nA)) < 1e-10


class TestNorm:
    """Tests for matrix/vector norms."""

    def test_frobenius_vector(self):
        """Frobenius norm of vector (same as 2-norm)."""
        a = rp.asarray([1.0, 2.0, 3.0])
        na = np.array([1.0, 2.0, 3.0])
        assert abs(rp.norm(a) - np.linalg.norm(na)) < 1e-10

    def test_frobenius_matrix(self):
        """Frobenius norm of matrix."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        nA = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert abs(rp.norm(A) - np.linalg.norm(nA, 'fro')) < 1e-10

    def test_frobenius_explicit(self):
        """Explicit fro argument."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        nA = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert abs(rp.norm(A, 'fro') - np.linalg.norm(nA, 'fro')) < 1e-10

    def test_identity_norm(self):
        """Frobenius norm of identity."""
        r = rp.norm(rp.eye(3))
        n = np.linalg.norm(np.eye(3), 'fro')
        assert abs(r - n) < 1e-10
