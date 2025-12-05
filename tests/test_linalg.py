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


class TestQR:
    """Tests for QR decomposition."""

    def test_square(self):
        """QR of square matrix."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        Q, R = rp.qr(A)

        nA = np.array([[1.0, 2.0], [3.0, 4.0]])
        nQ, nR = np.linalg.qr(nA)

        # Q @ R should reconstruct A
        assert_eq(Q @ R, A)

        # R should be upper triangular (lower part ~0)
        assert abs(float(R[1, 0])) < 1e-10

    def test_tall(self):
        """QR of tall matrix (m > n)."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Q, R = rp.qr(A)

        # Q should be (3, 2), R should be (2, 2)
        assert Q.shape == (3, 2)
        assert R.shape == (2, 2)

        # Q @ R should reconstruct A
        assert_eq(Q @ R, A)

    def test_wide(self):
        """QR of wide matrix (m < n)."""
        A = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Q, R = rp.qr(A)

        # Q should be (2, 2), R should be (2, 3)
        assert Q.shape == (2, 2)
        assert R.shape == (2, 3)

        # Q @ R should reconstruct A
        assert_eq(Q @ R, A)

    def test_orthogonal_q(self):
        """Q should be orthogonal (Q^T Q = I)."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Q, R = rp.qr(A)

        # Q^T @ Q should be identity
        QtQ = Q.T @ Q
        assert_eq(QtQ, rp.eye(2))


class TestSVD:
    """Tests for SVD decomposition."""

    def test_square(self):
        """SVD of square matrix."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        U, S, Vt = rp.svd(A)

        # U @ diag(S) @ Vt should reconstruct A
        # Build diagonal matrix from S
        S_diag = rp.asarray([[S[0], 0.0], [0.0, S[1]]])
        reconstructed = U @ S_diag @ Vt
        assert_eq(reconstructed, A)

    def test_tall(self):
        """SVD of tall matrix."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        U, S, Vt = rp.svd(A)

        # Shapes: U(3,2), S(2,), Vt(2,2)
        assert U.shape == (3, 2)
        assert S.shape == (2,)
        assert Vt.shape == (2, 2)

        # Reconstruct
        S_diag = rp.asarray([[S[0], 0.0], [0.0, S[1]]])
        reconstructed = U @ S_diag @ Vt
        assert_eq(reconstructed, A)

    def test_wide(self):
        """SVD of wide matrix."""
        A = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        U, S, Vt = rp.svd(A)

        # Shapes: U(2,2), S(2,), Vt(2,3)
        assert U.shape == (2, 2)
        assert S.shape == (2,)
        assert Vt.shape == (2, 3)

    def test_singular_values_positive(self):
        """Singular values should be non-negative."""
        A = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        U, S, Vt = rp.svd(A)
        assert float(S[0]) >= 0
        assert float(S[1]) >= 0

    def test_singular_values_descending(self):
        """Singular values should be in descending order."""
        A = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        U, S, Vt = rp.svd(A)
        assert float(S[0]) >= float(S[1])
        assert float(S[1]) >= float(S[2])
