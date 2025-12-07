"""Tests for matrix multiplication (matmul)."""

import numpy as np
import rumpy as rp

from helpers import assert_eq


class TestMatmul2D:
    """Basic 2D matrix multiplication."""

    def test_square(self):
        """Square matrices."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)

    def test_rectangular(self):
        """Non-square matrices."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
        b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)

    def test_matmul_function(self):
        """Test rp.matmul() function."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])

        r = rp.matmul(rp.asarray(a), rp.asarray(b))
        n = np.matmul(a, b)

        assert_eq(r, n)


class TestMatmul1D:
    """1D vector multiplications."""

    def test_1d_1d(self):
        """Inner product: (n,) @ (n,) -> scalar."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        # Result should be scalar (or squeezed to near-scalar)
        r_np = np.asarray(r)
        assert r_np.ravel()[0] == n

    def test_1d_2d(self):
        """Vector-matrix: (n,) @ (n, p) -> (p,)."""
        a = np.array([1.0, 2.0, 3.0])  # (3,)
        b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)

    def test_2d_1d(self):
        """Matrix-vector: (m, n) @ (n,) -> (m,)."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        b = np.array([1.0, 2.0, 3.0])  # (3,)

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)


class TestMatmulBatched:
    """Batched matrix multiplication."""

    def test_batched_same_shape(self):
        """Same batch shape: (B, M, N) @ (B, N, P) -> (B, M, P)."""
        a = np.random.randn(3, 2, 4)
        b = np.random.randn(3, 4, 5)

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)

    def test_batched_broadcast(self):
        """Broadcasting: (1, M, N) @ (B, N, P) -> (B, M, P)."""
        a = np.random.randn(1, 2, 4)
        b = np.random.randn(3, 4, 5)

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)

    def test_2d_vs_3d(self):
        """2D broadcast with 3D: (M, N) @ (B, N, P) -> (B, M, P)."""
        a = np.random.randn(2, 4)
        b = np.random.randn(3, 4, 5)

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)


class TestMatmulEdgeCases:
    """Edge cases."""

    def test_identity(self):
        """Multiply by identity matrix."""
        a = np.random.randn(3, 3)
        eye = np.eye(3)

        r = rp.asarray(a) @ rp.asarray(eye)
        assert_eq(r, a)

    def test_1x1(self):
        """1x1 matrices."""
        a = np.array([[2.0]])
        b = np.array([[3.0]])

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)


import pytest


class TestMatmulDtypes:
    """Test matmul with different dtypes."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_float_dtypes(self, dtype):
        """Test float32 and float64."""
        a = np.random.randn(50, 30).astype(dtype)
        b = np.random.randn(30, 40).astype(dtype)

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex64 and complex128."""
        a = (np.random.randn(50, 30) + 1j * np.random.randn(50, 30)).astype(dtype)
        b = (np.random.randn(30, 40) + 1j * np.random.randn(30, 40)).astype(dtype)

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        # complex64 has lower precision (~1e-5 tolerance)
        rtol = 1e-5 if dtype == np.complex64 else 1e-7
        assert_eq(r, n, rtol=rtol)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_square_matrices(self, dtype):
        """Test square matrices with each dtype."""
        if np.issubdtype(dtype, np.complexfloating):
            a = (np.random.randn(100, 100) + 1j * np.random.randn(100, 100)).astype(dtype)
            b = (np.random.randn(100, 100) + 1j * np.random.randn(100, 100)).astype(dtype)
        else:
            a = np.random.randn(100, 100).astype(dtype)
            b = np.random.randn(100, 100).astype(dtype)

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        # float32/complex64 have lower precision (complex64 accumulates more error)
        rtol = 2e-5 if dtype in (np.float32, np.complex64) else 1e-7
        assert_eq(r, n, rtol=rtol)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_1d_matmul(self, dtype):
        """Test 1D @ 2D and 2D @ 1D with different dtypes."""
        # 1D @ 2D
        a = np.random.randn(30).astype(dtype)
        b = np.random.randn(30, 40).astype(dtype)
        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b
        assert_eq(r, n)

        # 2D @ 1D
        a = np.random.randn(50, 30).astype(dtype)
        b = np.random.randn(30).astype(dtype)
        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b
        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_integer_dtypes(self, dtype):
        """Test integer matmul."""
        a = np.random.randint(-10, 10, size=(50, 30)).astype(dtype)
        b = np.random.randint(-10, 10, size=(30, 40)).astype(dtype)

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)

    def test_bool_dtype(self):
        """Test boolean matmul (AND/OR semiring)."""
        a = np.random.randint(0, 2, size=(50, 30)).astype(bool)
        b = np.random.randint(0, 2, size=(30, 40)).astype(bool)

        r = rp.asarray(a) @ rp.asarray(b)
        n = a @ b

        assert_eq(r, n)
