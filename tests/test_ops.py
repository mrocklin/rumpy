"""Tests for element-wise operations."""

import numpy as np
import pytest

import rumpy
from helpers import assert_eq


class TestBinaryOpsArray:
    """Test binary operations between arrays (same shape)."""

    def test_add(self):
        r1 = rumpy.arange(10)
        r2 = rumpy.arange(10, 20)
        n1 = np.arange(10, dtype=np.float64)
        n2 = np.arange(10, 20, dtype=np.float64)
        assert_eq(r1 + r2, n1 + n2)

    def test_sub(self):
        r1 = rumpy.arange(10)
        r2 = rumpy.arange(10, 20)
        n1 = np.arange(10, dtype=np.float64)
        n2 = np.arange(10, 20, dtype=np.float64)
        assert_eq(r1 - r2, n1 - n2)

    def test_mul(self):
        r1 = rumpy.arange(10)
        r2 = rumpy.arange(10, 20)
        n1 = np.arange(10, dtype=np.float64)
        n2 = np.arange(10, 20, dtype=np.float64)
        assert_eq(r1 * r2, n1 * n2)

    def test_div(self):
        r1 = rumpy.arange(1, 11)
        r2 = rumpy.arange(1, 11)
        n1 = np.arange(1, 11, dtype=np.float64)
        n2 = np.arange(1, 11, dtype=np.float64)
        assert_eq(r1 / r2, n1 / n2)

    def test_2d_add(self):
        r1 = rumpy.arange(12).reshape([3, 4])
        r2 = rumpy.arange(12, 24).reshape([3, 4])
        n1 = np.arange(12, dtype=np.float64).reshape(3, 4)
        n2 = np.arange(12, 24, dtype=np.float64).reshape(3, 4)
        assert_eq(r1 + r2, n1 + n2)

    def test_ops_on_view(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[2:8] + r[2:8], n[2:8] + n[2:8])

    def test_shape_mismatch_raises(self):
        r1 = rumpy.arange(10)
        r2 = rumpy.arange(5)
        with pytest.raises(ValueError):
            r1 + r2


class TestBinaryOpsScalar:
    """Test binary operations with scalars."""

    def test_add_scalar(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r + 5, n + 5)

    def test_radd_scalar(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(5 + r, 5 + n)

    def test_sub_scalar(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r - 3, n - 3)

    def test_rsub_scalar(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(10 - r, 10 - n)

    def test_mul_scalar(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r * 2, n * 2)

    def test_rmul_scalar(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(2 * r, 2 * n)

    def test_div_scalar(self):
        r = rumpy.arange(1, 11)
        n = np.arange(1, 11, dtype=np.float64)
        assert_eq(r / 2, n / 2)

    def test_rdiv_scalar(self):
        r = rumpy.arange(1, 11)
        n = np.arange(1, 11, dtype=np.float64)
        assert_eq(10 / r, 10 / n)


class TestUnaryOps:
    """Test unary operations."""

    def test_neg(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(-r, -n)

    def test_neg_2d(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(-r, -n)

    def test_abs(self):
        r = rumpy.arange(-5, 5)
        n = np.arange(-5, 5, dtype=np.float64)
        assert_eq(abs(r), abs(n))

    def test_abs_view(self):
        r = rumpy.arange(-10, 10)
        n = np.arange(-10, 10, dtype=np.float64)
        assert_eq(abs(r[5:15]), abs(n[5:15]))


class TestDtypePromotion:
    """Test that dtype promotion works correctly."""

    def test_int_float_promotion(self):
        r_int = rumpy.arange(10, dtype="int64")
        r_float = rumpy.arange(10, dtype="float64")
        n_int = np.arange(10, dtype=np.int64)
        n_float = np.arange(10, dtype=np.float64)
        # Result should be float64
        result = r_int + r_float
        expected = n_int + n_float
        assert_eq(result, expected)

    def test_float32_float64_promotion(self):
        r32 = rumpy.arange(10, dtype="float32")
        r64 = rumpy.arange(10, dtype="float64")
        n32 = np.arange(10, dtype=np.float32)
        n64 = np.arange(10, dtype=np.float64)
        result = r32 + r64
        expected = n32 + n64
        assert_eq(result, expected)
