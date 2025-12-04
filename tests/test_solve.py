"""Tests for linear system solver."""

import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestSolve:
    """Tests for solve(A, b) -> x where Ax = b."""

    def test_2x2(self):
        """Simple 2x2 system."""
        A = rp.asarray([[3.0, 1.0], [1.0, 2.0]])
        b = rp.asarray([9.0, 8.0])
        x = rp.solve(A, b)

        nA = np.array([[3.0, 1.0], [1.0, 2.0]])
        nb = np.array([9.0, 8.0])
        nx = np.linalg.solve(nA, nb)
        assert_eq(x, nx)

    def test_3x3(self):
        """3x3 system."""
        A = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        b = rp.asarray([1.0, 2.0, 3.0])
        x = rp.solve(A, b)

        nA = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        nb = np.array([1.0, 2.0, 3.0])
        nx = np.linalg.solve(nA, nb)
        assert_eq(x, nx)

    def test_multiple_rhs(self):
        """Multiple right-hand sides (b is 2D)."""
        A = rp.asarray([[2.0, 1.0], [1.0, 3.0]])
        b = rp.asarray([[1.0, 2.0], [3.0, 4.0]])  # (2, 2) - two RHS
        x = rp.solve(A, b)

        nA = np.array([[2.0, 1.0], [1.0, 3.0]])
        nb = np.array([[1.0, 2.0], [3.0, 4.0]])
        nx = np.linalg.solve(nA, nb)
        assert_eq(x, nx)

    def test_identity(self):
        """Identity matrix - x should equal b."""
        A = rp.eye(3)
        b = rp.asarray([1.0, 2.0, 3.0])
        x = rp.solve(A, b)
        assert_eq(x, b)

    def test_verify_solution(self):
        """Verify Ax = b."""
        A = rp.asarray([[4.0, 3.0], [6.0, 3.0]])
        b = rp.asarray([10.0, 12.0])
        x = rp.solve(A, b)
        # Check that A @ x ≈ b
        result = A @ x
        assert_eq(result, b)
