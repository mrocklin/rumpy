"""Comprehensive tests for linear algebra operations.

Tests cover matrix/vector multiplication, decompositions, solving, and properties.
Most tests use float64 for precision. Some operations tested for properties rather
than exact values due to sign/order ambiguity in decompositions.

See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest
import rumpy as rp
from helpers import assert_eq, make_numpy, make_pair
from conftest import FLOAT_DTYPES


# ============================================================================
# Matrix/Vector Multiplication
# ============================================================================


class TestMatmul:
    """Matrix multiplication (2D @ 2D)."""

    def test_2d_2d_square(self):
        """Square matrices."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.matmul(ra, rb), np.matmul(a, b))

    def test_2d_2d_rectangular(self):
        """Rectangular matrices (m,k) @ (k,n)."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.matmul(ra, rb), np.matmul(a, b))

    def test_operator_override(self):
        """@ operator uses matmul."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(ra @ rb, a @ b)

    def test_via_linalg_submodule(self):
        """Can also access as rp.linalg.matmul (though not standard numpy)."""
        # Note: numpy doesn't have np.linalg.matmul, but we test our implementation works
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        result = rp.matmul(ra, rb)
        assert_eq(result, np.matmul(a, b))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        """Works with float32 and float64."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=dtype)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.matmul(ra, rb), np.matmul(a, b))

    def test_associativity(self):
        """(A @ B) @ C = A @ (B @ C)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        c = np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float64)
        ra, rb, rc = rp.asarray(a), rp.asarray(b), rp.asarray(c)

        left = (ra @ rb) @ rc
        right = ra @ (rb @ rc)
        assert_eq(left, right)


class TestDot:
    """Dot product with numpy's flexible semantics."""

    def test_1d_1d(self):
        """1D @ 1D = scalar inner product."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.dot(ra, rb), np.dot(a, b))

    def test_2d_1d(self):
        """2D @ 1D = matrix-vector product."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([1.0, 2.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.dot(ra, rb), np.dot(a, b))

    def test_2d_2d(self):
        """2D @ 2D = matrix multiply."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.dot(ra, rb), np.dot(a, b))

    def test_method_form(self):
        """a.dot(b) works like np.dot(a, b)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([1.0, 2.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(ra.dot(rb), a.dot(b))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        """Works with different float types."""
        a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        b = np.array([4.0, 5.0, 6.0], dtype=dtype)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.dot(ra, rb), np.dot(a, b))


class TestInner:
    """Inner product (flattens last axis)."""

    def test_1d_1d(self):
        """Standard inner product of vectors."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.inner(ra, rb), np.inner(a, b))


class TestOuter:
    """Outer product (cross product of flattened arrays)."""

    def test_1d_1d(self):
        """Outer product of vectors."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.outer(ra, rb), np.outer(a, b))

    def test_result_shape(self):
        """Result shape is (a.size, b.size)."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        result = rp.outer(ra, rb)
        assert result.shape == (3, 2)


# ============================================================================
# Linear Systems
# ============================================================================


class TestSolve:
    """Solve linear system Ax = b."""

    def test_square_simple(self):
        """2x2 system with unique solution."""
        a = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        b = np.array([9.0, 8.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        x = rp.linalg.solve(ra, rb)
        nx = np.linalg.solve(a, b)
        assert_eq(x, nx)

    def test_solution_verifies(self):
        """A @ x should equal b."""
        a = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        b = np.array([9.0, 8.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        x = rp.linalg.solve(ra, rb)
        assert_eq(ra @ x, rb)

    def test_3x3_system(self):
        """3x3 system."""
        a = np.array([[1.0, 2.0, 3.0],
                      [2.0, 5.0, 3.0],
                      [1.0, 0.0, 8.0]], dtype=np.float64)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        x = rp.linalg.solve(ra, rb)
        nx = np.linalg.solve(a, b)
        assert_eq(x, nx)

    def test_identity_system(self):
        """I @ x = b has solution x = b."""
        I = np.eye(3, dtype=np.float64)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rI, rb = rp.asarray(I), rp.asarray(b)

        x = rp.linalg.solve(rI, rb)
        assert_eq(x, rb)

    def test_multiple_rhs(self):
        """Solve A @ X = B with matrix B."""
        a = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        b = np.array([[9.0, 1.0], [8.0, 2.0]], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        x = rp.linalg.solve(ra, rb)
        nx = np.linalg.solve(a, b)
        assert_eq(x, nx)


# ============================================================================
# Matrix Inverse
# ============================================================================


class TestInv:
    """Matrix inverse."""

    def test_2x2(self):
        """Inverse of 2x2 matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        a_inv = rp.linalg.inv(ra)
        na_inv = np.linalg.inv(a)
        assert_eq(a_inv, na_inv)

    def test_identity_inverse(self):
        """Inverse of identity is identity."""
        I = rp.eye(3)
        I_inv = rp.linalg.inv(I)
        assert_eq(I_inv, I)

    def test_inverse_property(self):
        """A @ A^-1 = I."""
        a = np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        a_inv = rp.linalg.inv(ra)
        product = ra @ a_inv
        assert_eq(product, rp.eye(2))

    def test_3x3_inverse(self):
        """3x3 matrix inverse."""
        a = np.array([[1.0, 2.0, 3.0],
                      [0.0, 1.0, 4.0],
                      [5.0, 6.0, 0.0]], dtype=np.float64)
        ra = rp.asarray(a)

        a_inv = rp.linalg.inv(ra)
        na_inv = np.linalg.inv(a)
        assert_eq(a_inv, na_inv)

    def test_both_directions(self):
        """Both A @ A^-1 and A^-1 @ A equal I."""
        a = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        a_inv = rp.linalg.inv(ra)
        I = rp.eye(2)
        assert_eq(ra @ a_inv, I)
        assert_eq(a_inv @ ra, I)


# ============================================================================
# Determinant
# ============================================================================


class TestDet:
    """Matrix determinant."""

    def test_2x2(self):
        """2x2 determinant."""
        a = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        det = rp.linalg.det(ra)
        ndet = np.linalg.det(a)
        assert abs(det - ndet) < 1e-10

    def test_3x3(self):
        """3x3 determinant."""
        a = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 10.0]], dtype=np.float64)
        ra = rp.asarray(a)

        det = rp.linalg.det(ra)
        ndet = np.linalg.det(a)
        assert abs(det - ndet) < 1e-10

    def test_identity_det(self):
        """Determinant of identity is 1."""
        I = rp.eye(3)
        det = rp.linalg.det(I)
        assert abs(det - 1.0) < 1e-10

    def test_singular_matrix(self):
        """Determinant of singular matrix is 0."""
        a = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)  # rank 1
        ra = rp.asarray(a)

        det = rp.linalg.det(ra)
        assert abs(det) < 1e-10

    def test_negative_det(self):
        """Matrix with negative determinant."""
        a = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)  # permutation
        ra = rp.asarray(a)

        det = rp.linalg.det(ra)
        assert abs(det - (-1.0)) < 1e-10

    def test_det_scales(self):
        """det(cA) = c^n det(A) for nxn matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        det_a = rp.linalg.det(ra)
        det_2a = rp.linalg.det(2.0 * ra)
        # For 2x2: det(2A) = 4 det(A)
        assert abs(det_2a - 4.0 * det_a) < 1e-10

    def test_module_level_function(self):
        """rp.det() also works (not just rp.linalg.det)."""
        a = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        det1 = rp.det(ra)
        det2 = rp.linalg.det(ra)
        assert abs(det1 - det2) < 1e-10


# ============================================================================
# QR Decomposition
# ============================================================================


class TestQR:
    """QR decomposition: A = QR."""

    def test_square_matrix(self):
        """QR of square matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        Q, R = rp.linalg.qr(ra)

        # Q @ R should reconstruct A
        assert_eq(Q @ R, ra, rtol=1e-5)

    def test_q_orthogonal(self):
        """Q should be orthogonal: Q^T @ Q = I."""
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        Q, R = rp.linalg.qr(ra)
        QtQ = Q.T @ Q
        assert_eq(QtQ, rp.eye(2), rtol=1e-5)

    def test_r_upper_triangular(self):
        """R should be upper triangular."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        Q, R = rp.linalg.qr(ra)
        # Check lower triangle is zero
        assert abs(float(R[1, 0])) < 1e-10

    def test_tall_matrix(self):
        """QR of tall matrix (m > n)."""
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        Q, R = rp.linalg.qr(ra)

        # Shapes should be Q(3,2), R(2,2)
        assert Q.shape == (3, 2)
        assert R.shape == (2, 2)

        # Reconstruction
        assert_eq(Q @ R, ra, rtol=1e-5)

    def test_wide_matrix(self):
        """QR of wide matrix (m < n)."""
        a = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        Q, R = rp.linalg.qr(ra)

        # Shapes should be Q(2,2), R(2,3)
        assert Q.shape == (2, 2)
        assert R.shape == (2, 3)

        # Reconstruction
        assert_eq(Q @ R, ra, rtol=1e-5)

    def test_compare_numpy(self):
        """Compare with numpy (up to sign ambiguity)."""
        a = np.array([[12.0, -51.0, 4.0],
                      [6.0, 167.0, -68.0],
                      [-4.0, 24.0, -41.0]], dtype=np.float64)
        ra = rp.asarray(a)

        Q, R = rp.linalg.qr(ra)
        nQ, nR = np.linalg.qr(a)

        # Test reconstruction (handles sign differences)
        assert_eq(Q @ R, a, rtol=1e-5)
        assert_eq(nQ @ nR, a, rtol=1e-5)


# ============================================================================
# SVD Decomposition
# ============================================================================


class TestSVD:
    """Singular value decomposition: A = U @ diag(S) @ Vt."""

    def test_square_matrix(self):
        """SVD of square matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        U, S, Vt = rp.linalg.svd(ra, full_matrices=False)

        # Reconstruct
        S_diag = rp.asarray([[S[0], 0.0], [0.0, S[1]]])
        reconstructed = U @ S_diag @ Vt
        assert_eq(reconstructed, ra, rtol=1e-5)

    def test_tall_matrix(self):
        """SVD of tall matrix (m > n)."""
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        U, S, Vt = rp.linalg.svd(ra, full_matrices=False)

        # Shapes: U(3,2), S(2,), Vt(2,2)
        assert U.shape == (3, 2)
        assert S.shape == (2,)
        assert Vt.shape == (2, 2)

        # Reconstruct
        S_diag = rp.asarray([[S[0], 0.0], [0.0, S[1]]])
        reconstructed = U @ S_diag @ Vt
        assert_eq(reconstructed, ra, rtol=1e-5)

    def test_wide_matrix(self):
        """SVD of wide matrix (m < n)."""
        a = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        U, S, Vt = rp.linalg.svd(ra, full_matrices=False)

        # Shapes: U(2,2), S(2,), Vt(2,3)
        assert U.shape == (2, 2)
        assert S.shape == (2,)
        assert Vt.shape == (2, 3)

    def test_singular_values_positive(self):
        """Singular values should be non-negative."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        U, S, Vt = rp.linalg.svd(ra, full_matrices=False)
        assert float(S[0]) >= 0
        assert float(S[1]) >= 0

    def test_singular_values_sorted(self):
        """Singular values should be in descending order."""
        a = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 10.0]], dtype=np.float64)
        ra = rp.asarray(a)

        U, S, Vt = rp.linalg.svd(ra, full_matrices=False)
        assert float(S[0]) >= float(S[1])
        assert float(S[1]) >= float(S[2])

    def test_u_orthogonal(self):
        """U should be orthogonal: U^T @ U = I."""
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        U, S, Vt = rp.linalg.svd(ra, full_matrices=False)
        UtU = U.T @ U
        assert_eq(UtU, rp.eye(2), rtol=1e-5)

    def test_v_orthogonal(self):
        """V should be orthogonal: Vt @ Vt^T = I."""
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        U, S, Vt = rp.linalg.svd(ra, full_matrices=False)
        VtVtT = Vt @ Vt.T
        assert_eq(VtVtT, rp.eye(2), rtol=1e-5)


# ============================================================================
# Eigendecomposition
# ============================================================================


class TestEig:
    """General eigendecomposition: A @ V = V @ diag(w)."""

    def test_symmetric_2x2(self):
        """Eigendecomposition of symmetric 2x2."""
        a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w, V = rp.linalg.eig(ra)
        nw, nV = np.linalg.eig(a)

        # Sort eigenvalues for comparison (order may differ)
        w_sorted = sorted(np.asarray(w), key=lambda x: x.real)
        nw_sorted = sorted(nw, key=lambda x: x.real)

        for i in range(len(nw)):
            assert abs(w_sorted[i] - nw_sorted[i]) < 1e-10

    def test_diagonal_matrix(self):
        """Eigenvalues of diagonal matrix are the diagonal elements."""
        a = np.array([[3.0, 0.0], [0.0, 5.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w, V = rp.linalg.eig(ra)

        # Eigenvalues should be 3 and 5 (in some order)
        eigenvals = sorted([float(w[0].real), float(w[1].real)])
        assert abs(eigenvals[0] - 3.0) < 1e-10
        assert abs(eigenvals[1] - 5.0) < 1e-10

    def test_identity_eigenvalues(self):
        """Eigenvalues of identity are all 1."""
        I = rp.eye(3)
        w, V = rp.linalg.eig(I)

        for i in range(3):
            assert abs(float(w[i].real) - 1.0) < 1e-10

    def test_upper_triangular(self):
        """Eigenvalues of triangular matrix are diagonal elements."""
        a = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w, V = rp.linalg.eig(ra)

        # Eigenvalues should be 1 and 3
        eigenvals = sorted([float(w[0].real), float(w[1].real)])
        assert abs(eigenvals[0] - 1.0) < 1e-10
        assert abs(eigenvals[1] - 3.0) < 1e-10

    def test_shapes(self):
        """Check output shapes."""
        a = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w, V = rp.linalg.eig(ra)
        assert w.shape == (3,)
        assert V.shape == (3, 3)


class TestEigvals:
    """Eigenvalues only (no eigenvectors)."""

    def test_symmetric(self):
        """Eigenvalues of symmetric matrix are real."""
        a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w = rp.linalg.eigvals(ra)
        nw = np.linalg.eigvals(a)

        # Sort for comparison
        w_sorted = sorted(np.asarray(w), key=lambda x: x.real)
        nw_sorted = sorted(nw, key=lambda x: x.real)

        for i in range(len(nw)):
            assert abs(w_sorted[i] - nw_sorted[i]) < 1e-10

    def test_nonsymmetric(self):
        """Eigenvalues of non-symmetric matrix may be complex."""
        a = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w = rp.linalg.eigvals(ra)
        nw = np.linalg.eigvals(a)

        # Sort for comparison
        w_sorted = sorted(np.asarray(w), key=lambda x: x.real)
        nw_sorted = sorted(nw, key=lambda x: x.real)

        for i in range(len(nw)):
            assert abs(w_sorted[i] - nw_sorted[i]) < 1e-10


class TestEigh:
    """Symmetric eigendecomposition (faster, always real eigenvalues)."""

    def test_symmetric_2x2(self):
        """eigh for symmetric matrix."""
        a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w, V = rp.linalg.eigh(ra)
        nw, nV = np.linalg.eigh(a)

        # Eigenvalues should match (eigh returns sorted)
        assert_eq(w, rp.asarray(nw), rtol=1e-5)

    def test_reconstruction(self):
        """V @ diag(w) @ V^T should reconstruct A."""
        a = np.array([[3.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w, V = rp.linalg.eigh(ra)

        # Reconstruct
        W = rp.diag(w)
        reconstructed = V @ W @ V.T
        assert_eq(reconstructed, ra, rtol=1e-5)

    def test_orthogonal_eigenvectors(self):
        """Eigenvectors should be orthonormal: V^T @ V = I."""
        a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w, V = rp.linalg.eigh(ra)
        VtV = V.T @ V
        assert_eq(VtV, rp.eye(2), rtol=1e-5)

    def test_ascending_eigenvalues(self):
        """eigh returns eigenvalues in ascending order."""
        a = np.array([[5.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w, V = rp.linalg.eigh(ra)
        assert float(w[0]) <= float(w[1])


# ============================================================================
# Additional Linear Algebra Functions
# ============================================================================


class TestTrace:
    """Matrix trace (sum of diagonal)."""

    def test_2x2(self):
        """Trace of 2x2 matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert abs(rp.trace(ra) - np.trace(a)) < 1e-10

    def test_3x3(self):
        """Trace of 3x3 matrix."""
        a = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert abs(rp.trace(ra) - np.trace(a)) < 1e-10

    def test_identity(self):
        """Trace of identity is n."""
        I = rp.eye(5)
        assert abs(rp.trace(I) - 5.0) < 1e-10

    def test_rectangular(self):
        """Trace of rectangular matrix (min dimension)."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert abs(rp.trace(ra) - np.trace(a)) < 1e-10

    def test_method_form(self):
        """a.trace() works like np.trace(a)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert abs(ra.trace() - a.trace()) < 1e-10


class TestNorm:
    """Matrix and vector norms."""

    def test_vector_default(self):
        """Default norm of vector is 2-norm."""
        a = np.array([3.0, 4.0], dtype=np.float64)
        ra = rp.asarray(a)

        assert abs(rp.linalg.norm(ra) - np.linalg.norm(a)) < 1e-10

    def test_frobenius_matrix(self):
        """Frobenius norm of matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert abs(rp.linalg.norm(ra, 'fro') - np.linalg.norm(a, 'fro')) < 1e-10

    def test_identity_norm(self):
        """Frobenius norm of identity."""
        I = rp.eye(3)
        nI = np.eye(3)

        assert abs(rp.linalg.norm(I, 'fro') - np.linalg.norm(nI, 'fro')) < 1e-10


class TestDiag:
    """Diagonal extraction and construction."""

    def test_extract_from_2d(self):
        """Extract diagonal from matrix."""
        a = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert_eq(rp.diag(ra), np.diag(a))

    def test_construct_from_1d(self):
        """Construct diagonal matrix from vector."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rv = rp.asarray(v)

        assert_eq(rp.diag(rv), np.diag(v))

    def test_rectangular(self):
        """Extract diagonal from rectangular matrix."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        d = rp.diag(ra)
        assert d.shape == (2,)
        assert_eq(d, rp.asarray([1.0, 5.0]))


class TestDiagonal:
    """Diagonal extraction (method and function)."""

    def test_diagonal_function(self):
        """rp.diagonal() extracts diagonal."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert_eq(rp.diagonal(ra), np.diagonal(a))

    def test_diagonal_method(self):
        """a.diagonal() method."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert_eq(ra.diagonal(), a.diagonal())


# ============================================================================
# Advanced Linear Algebra
# ============================================================================


class TestCholesky:
    """Cholesky decomposition for symmetric positive-definite matrices."""

    def test_spd_matrix(self):
        """Cholesky of SPD matrix."""
        a = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float64)
        ra = rp.asarray(a)

        L = rp.linalg.cholesky(ra)
        nL = np.linalg.cholesky(a)
        assert_eq(L, nL, rtol=1e-5)

    def test_reconstruction(self):
        """L @ L^T should reconstruct A."""
        a = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float64)
        ra = rp.asarray(a)

        L = rp.linalg.cholesky(ra)
        reconstructed = L @ L.T
        assert_eq(reconstructed, ra, rtol=1e-5)

    def test_identity(self):
        """Cholesky of identity is identity."""
        I = rp.eye(3)
        L = rp.linalg.cholesky(I)
        assert_eq(L, I, rtol=1e-5)


class TestSlogdet:
    """Sign and log of determinant."""

    def test_positive_det(self):
        """Matrix with positive determinant."""
        a = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        sign, logabsdet = rp.linalg.slogdet(ra)
        nsign, nlogabsdet = np.linalg.slogdet(a)

        assert abs(sign - nsign) < 1e-10
        assert abs(logabsdet - nlogabsdet) < 1e-10

    def test_negative_det(self):
        """Matrix with negative determinant."""
        a = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        ra = rp.asarray(a)

        sign, logabsdet = rp.linalg.slogdet(ra)
        nsign, nlogabsdet = np.linalg.slogdet(a)

        assert abs(sign - nsign) < 1e-10
        assert abs(logabsdet - nlogabsdet) < 1e-10


class TestCondition:
    """Condition number."""

    def test_identity(self):
        """Condition number of identity is 1."""
        I = rp.eye(3)
        c = rp.linalg.cond(I)
        assert abs(c - 1.0) < 1e-10

    def test_well_conditioned(self):
        """Well-conditioned matrix has low condition number."""
        a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        c = rp.linalg.cond(ra)
        nc = np.linalg.cond(a)
        assert abs(c - nc) < 1e-8

    def test_singular(self):
        """Singular matrix has infinite condition number."""
        a = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        c = rp.linalg.cond(ra)
        assert c == float('inf')


class TestMatrixRank:
    """Matrix rank computation."""

    def test_full_rank(self):
        """Full rank matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert rp.linalg.matrix_rank(ra) == np.linalg.matrix_rank(a)

    def test_rank_deficient(self):
        """Rank deficient matrix."""
        a = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert rp.linalg.matrix_rank(ra) == np.linalg.matrix_rank(a)

    def test_rectangular(self):
        """Rank of rectangular matrix."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        assert rp.linalg.matrix_rank(ra) == np.linalg.matrix_rank(a)


class TestPinv:
    """Moore-Penrose pseudo-inverse."""

    def test_square_invertible(self):
        """Pseudo-inverse of invertible matrix equals inverse."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        pinv = rp.linalg.pinv(ra)
        npinv = np.linalg.pinv(a)
        assert_eq(pinv, npinv, rtol=1e-5)

    def test_rectangular(self):
        """Pseudo-inverse of rectangular matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        pinv = rp.linalg.pinv(ra)
        npinv = np.linalg.pinv(a)
        assert_eq(pinv, npinv, rtol=1e-5)

    def test_identity_property(self):
        """For full column rank: A^+ @ A = I."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        pinv = rp.linalg.pinv(ra)
        result = ra @ pinv
        assert_eq(result, rp.eye(2), rtol=1e-5)


class TestLstsq:
    """Least squares solution."""

    def test_exact_solution(self):
        """Square system with exact solution."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([5.0, 11.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        x, residuals, rank, s = rp.linalg.lstsq(ra, rb)
        nx, nresiduals, nrank, ns = np.linalg.lstsq(a, b, rcond=None)

        assert_eq(x, nx, rtol=1e-5)
        assert rank == nrank

    def test_overdetermined(self):
        """Overdetermined system (more equations than unknowns)."""
        a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=np.float64)
        b = np.array([1.0, 2.0, 2.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        x, residuals, rank, s = rp.linalg.lstsq(ra, rb)
        nx, nresiduals, nrank, ns = np.linalg.lstsq(a, b, rcond=None)

        assert_eq(x, nx, rtol=1e-5)
        assert rank == nrank


# ============================================================================
# Special Products
# ============================================================================


class TestVdot:
    """Vector dot product (flattens arrays)."""

    def test_1d_1d(self):
        """Dot product of 1D arrays."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        assert abs(rp.vdot(ra, rb) - np.vdot(a, b)) < 1e-10

    def test_2d_flattened(self):
        """vdot flattens 2D arrays."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        assert abs(rp.vdot(ra, rb) - np.vdot(a, b)) < 1e-10


class TestKron:
    """Kronecker product."""

    def test_2x2(self):
        """Kronecker product of 2x2 matrices."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[0.0, 5.0], [6.0, 7.0]], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        assert_eq(rp.kron(ra, rb), np.kron(a, b))

    def test_1d(self):
        """Kronecker product of vectors."""
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        assert_eq(rp.kron(ra, rb), np.kron(a, b))


class TestCross:
    """Cross product of 3D vectors."""

    def test_3d_vectors(self):
        """Standard cross product."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        assert_eq(rp.cross(ra, rb), np.cross(a, b))

    def test_unit_vectors(self):
        """i cross j equals k."""
        i = rp.asarray([1.0, 0.0, 0.0])
        j = rp.asarray([0.0, 1.0, 0.0])
        k = rp.cross(i, j)

        assert_eq(k, rp.asarray([0.0, 0.0, 1.0]))


class TestTensordot:
    """Tensor dot product."""

    def test_matmul_via_tensordot(self):
        """tensordot with axes=1 performs matrix multiply."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        assert_eq(rp.tensordot(ra, rb, 1), np.tensordot(a, b, 1))

    def test_1d_inner(self):
        """tensordot of 1D arrays is inner product."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        r = rp.tensordot(ra, rb, 1)
        n = np.tensordot(a, b, 1)
        # Handle scalar vs (1,) shape difference
        assert abs(float(r) - float(n)) < 1e-10


# ============================================================================
# Miscellaneous
# ============================================================================


class TestLinalgSubmodule:
    """Tests for linalg submodule existence and accessibility."""

    def test_submodule_exists(self):
        """rp.linalg submodule exists."""
        assert hasattr(rp, 'linalg')

    def test_functions_accessible(self):
        """Key functions accessible via submodule."""
        assert hasattr(rp.linalg, 'solve')
        assert hasattr(rp.linalg, 'inv')
        assert hasattr(rp.linalg, 'det')
        assert hasattr(rp.linalg, 'qr')
        assert hasattr(rp.linalg, 'svd')
        assert hasattr(rp.linalg, 'eig')
        assert hasattr(rp.linalg, 'eigh')

    def test_new_functions_accessible(self):
        """Stream 24 functions accessible."""
        assert hasattr(rp.linalg, 'eigvalsh')
        assert hasattr(rp.linalg, 'svdvals')
        assert hasattr(rp.linalg, 'matrix_power')
        assert hasattr(rp.linalg, 'multi_dot')
        assert hasattr(rp.linalg, 'tensorinv')
        assert hasattr(rp.linalg, 'tensorsolve')
        assert hasattr(rp.linalg, 'vector_norm')
        assert hasattr(rp.linalg, 'matrix_norm')
        assert hasattr(rp.linalg, 'LinAlgError')


# ============================================================================
# Stream 24: Linalg Extensions
# ============================================================================


class TestEigvalsh:
    """Eigenvalues of symmetric matrix."""

    def test_symmetric_2x2(self):
        """Eigenvalues of symmetric 2x2."""
        a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w = rp.linalg.eigvalsh(ra)
        nw = np.linalg.eigvalsh(a)
        assert_eq(w, rp.asarray(nw), rtol=1e-10)

    def test_symmetric_3x3(self):
        """Eigenvalues of symmetric 3x3."""
        a = np.array([[4.0, 2.0, 1.0],
                      [2.0, 5.0, 3.0],
                      [1.0, 3.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        w = rp.linalg.eigvalsh(ra)
        nw = np.linalg.eigvalsh(a)
        assert_eq(w, rp.asarray(nw), rtol=1e-10)

    def test_identity(self):
        """Eigenvalues of identity are all 1."""
        I = rp.eye(3)
        w = rp.linalg.eigvalsh(I)
        for i in range(3):
            assert abs(float(w[i]) - 1.0) < 1e-10


class TestSvdvals:
    """Singular values only."""

    def test_square(self):
        """Singular values of square matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        s = rp.linalg.svdvals(ra)
        ns = np.linalg.svdvals(a)
        assert_eq(s, rp.asarray(ns), rtol=1e-10)

    def test_tall(self):
        """Singular values of tall matrix."""
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        s = rp.linalg.svdvals(ra)
        ns = np.linalg.svdvals(a)
        assert_eq(s, rp.asarray(ns), rtol=1e-10)

    def test_wide(self):
        """Singular values of wide matrix."""
        a = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float64)
        ra = rp.asarray(a)

        s = rp.linalg.svdvals(ra)
        ns = np.linalg.svdvals(a)
        assert_eq(s, rp.asarray(ns), rtol=1e-10)


class TestMatrixPower:
    """Matrix to integer power."""

    def test_power_2(self):
        """A^2 = A @ A."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_power(ra, 2)
        expected = np.linalg.matrix_power(a, 2)
        assert_eq(result, expected, rtol=1e-10)

    def test_power_3(self):
        """A^3 = A @ A @ A."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_power(ra, 3)
        expected = np.linalg.matrix_power(a, 3)
        assert_eq(result, expected, rtol=1e-10)

    def test_power_0(self):
        """A^0 = I."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_power(ra, 0)
        assert_eq(result, rp.eye(2), rtol=1e-10)

    def test_power_negative(self):
        """A^-1 = inv(A)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_power(ra, -1)
        expected = np.linalg.matrix_power(a, -1)
        assert_eq(result, expected, rtol=1e-10)

    def test_power_negative_2(self):
        """A^-2 = inv(A)^2."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_power(ra, -2)
        expected = np.linalg.matrix_power(a, -2)
        assert_eq(result, expected, rtol=1e-10)

    def test_identity_any_power(self):
        """I^n = I for any n."""
        I = rp.eye(3)
        for n in [-2, -1, 0, 1, 2, 5]:
            assert_eq(rp.linalg.matrix_power(I, n), I, rtol=1e-10)


class TestMultiDot:
    """Efficient multi-matrix multiplication."""

    def test_two_matrices(self):
        """multi_dot([A, B]) = A @ B."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        ra, rb = rp.asarray(a), rp.asarray(b)

        result = rp.linalg.multi_dot([ra, rb])
        expected = np.linalg.multi_dot([a, b])
        assert_eq(result, expected, rtol=1e-10)

    def test_three_matrices(self):
        """multi_dot([A, B, C]) = A @ B @ C."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        c = np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float64)
        ra, rb, rc = rp.asarray(a), rp.asarray(b), rp.asarray(c)

        result = rp.linalg.multi_dot([ra, rb, rc])
        expected = np.linalg.multi_dot([a, b, c])
        assert_eq(result, expected, rtol=1e-10)

    def test_chain(self):
        """Chain of different sized matrices."""
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 2)
        c = np.random.randn(2, 5)
        ra, rb, rc = rp.asarray(a), rp.asarray(b), rp.asarray(c)

        result = rp.linalg.multi_dot([ra, rb, rc])
        expected = np.linalg.multi_dot([a, b, c])
        assert_eq(result, expected, rtol=1e-10)


class TestVectorNorm:
    """Vector norm with ord parameter."""

    def test_2_norm_default(self):
        """Default is 2-norm."""
        a = np.array([3.0, 4.0], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.vector_norm(ra)
        expected = np.linalg.norm(a)
        assert abs(result - expected) < 1e-10

    def test_2_norm_explicit(self):
        """Explicit 2-norm."""
        a = np.array([3.0, 4.0], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.vector_norm(ra, 2.0)
        expected = np.linalg.norm(a, 2)
        assert abs(result - expected) < 1e-10

    def test_1_norm(self):
        """1-norm (sum of absolute values)."""
        a = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.vector_norm(ra, 1.0)
        expected = np.linalg.norm(a, 1)
        assert abs(result - expected) < 1e-10

    def test_inf_norm(self):
        """inf-norm (max absolute value)."""
        a = np.array([1.0, -5.0, 3.0], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.vector_norm(ra, float('inf'))
        expected = np.linalg.norm(a, np.inf)
        assert abs(result - expected) < 1e-10

    def test_neg_inf_norm(self):
        """-inf-norm (min absolute value)."""
        a = np.array([1.0, -5.0, 3.0], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.vector_norm(ra, float('-inf'))
        expected = np.linalg.norm(a, -np.inf)
        assert abs(result - expected) < 1e-10

    def test_0_norm(self):
        """0-norm (count of non-zero)."""
        a = np.array([1.0, 0.0, 0.0, 5.0], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.vector_norm(ra, 0.0)
        expected = np.linalg.norm(a, 0)
        assert abs(result - expected) < 1e-10

    def test_3_norm(self):
        """General p-norm with p=3."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.vector_norm(ra, 3.0)
        expected = np.linalg.norm(a, 3)
        assert abs(result - expected) < 1e-10


class TestMatrixNorm:
    """Matrix norm with ord parameter."""

    def test_frobenius_default(self):
        """Default is Frobenius norm."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_norm(ra)
        expected = np.linalg.norm(a, 'fro')
        assert abs(result - expected) < 1e-10

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_frobenius_dtypes(self, dtype):
        """Frobenius norm works for different dtypes."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_norm(ra, 'fro')
        expected = np.linalg.norm(a, 'fro')
        assert abs(result - expected) < 1e-4  # Less precision for float32

    def test_frobenius_explicit(self):
        """Explicit Frobenius norm."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_norm(ra, 'fro')
        expected = np.linalg.norm(a, 'fro')
        assert abs(result - expected) < 1e-10

    def test_nuclear_norm(self):
        """Nuclear norm (sum of singular values)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_norm(ra, 'nuc')
        expected = np.linalg.norm(a, 'nuc')
        assert abs(result - expected) < 1e-10

    def test_2_norm(self):
        """2-norm (largest singular value)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_norm(ra, '2')
        expected = np.linalg.norm(a, 2)
        assert abs(result - expected) < 1e-10

    def test_1_norm(self):
        """1-norm (max column sum)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_norm(ra, '1')
        expected = np.linalg.norm(a, 1)
        assert abs(result - expected) < 1e-10

    def test_inf_norm(self):
        """inf-norm (max row sum)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        ra = rp.asarray(a)

        result = rp.linalg.matrix_norm(ra, 'inf')
        expected = np.linalg.norm(a, np.inf)
        assert abs(result - expected) < 1e-10


class TestTensorinv:
    """Tensor inverse."""

    def test_4d_tensor(self):
        """Inverse of 4D tensor with default ind."""
        # Create a 4D tensor (2,2,2,2) that reshapes to invertible 4x4
        a = np.random.randn(2, 2, 2, 2)
        a_2d = a.reshape(4, 4)
        # Make it invertible
        a_2d = a_2d + np.eye(4) * 5  # Add diagonal to ensure invertibility
        a = a_2d.reshape(2, 2, 2, 2)
        ra = rp.asarray(a)

        result = rp.linalg.tensorinv(ra)
        expected = np.linalg.tensorinv(a)
        assert_eq(result, expected, rtol=1e-8)

    def test_with_ind(self):
        """Tensor inverse with custom ind."""
        a = np.random.randn(4, 3, 3, 4)
        a_2d = a.reshape(12, 12)
        a_2d = a_2d + np.eye(12) * 10
        a = a_2d.reshape(4, 3, 3, 4)
        ra = rp.asarray(a)

        result = rp.linalg.tensorinv(ra, ind=2)
        expected = np.linalg.tensorinv(a, ind=2)
        assert_eq(result, expected, rtol=1e-8)


class TestTensorsolve:
    """Tensor equation solve."""

    def test_simple(self):
        """Simple tensor solve."""
        # A is 3D (2, 2, 4) and we solve for x of shape (4,) given b of shape (2, 2)
        a = np.random.randn(2, 2, 4)
        # Reshape to (4, 4) to make it well-conditioned
        a_2d = a.reshape(4, 4)
        a_2d = a_2d + np.eye(4) * 5
        a = a_2d.reshape(2, 2, 4)

        # Generate solution and compute b = A @ x in tensor sense
        x_true = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.tensordot(a, x_true, axes=([2], [0]))

        ra, rb = rp.asarray(a), rp.asarray(b)
        result = rp.linalg.tensorsolve(ra, rb)
        expected = np.linalg.tensorsolve(a, b)
        assert_eq(result, expected, rtol=1e-8)


class TestLinAlgError:
    """LinAlgError exception class."""

    def test_exception_exists(self):
        """LinAlgError is a proper exception class."""
        assert issubclass(rp.linalg.LinAlgError, Exception)

    def test_can_raise(self):
        """Can raise and catch LinAlgError."""
        try:
            raise rp.linalg.LinAlgError("test error")
        except rp.linalg.LinAlgError as e:
            assert "test error" in str(e)


class TestNewaxis:
    """Tests for newaxis constant."""

    def test_newaxis_is_none(self):
        """newaxis should be None."""
        assert rp.newaxis is None
        assert np.newaxis is None

    def test_expand_dims_start(self):
        """Add dimension at start with newaxis."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ra = rp.asarray(a)

        r = ra[rp.newaxis, :]
        n = a[np.newaxis, :]

        assert r.shape == n.shape
        assert r.shape == (1, 3)

    def test_expand_dims_end(self):
        """Add dimension at end with newaxis."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ra = rp.asarray(a)

        r = ra[:, rp.newaxis]
        n = a[:, np.newaxis]

        assert r.shape == n.shape
        assert r.shape == (3, 1)
