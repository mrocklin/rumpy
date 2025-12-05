"""Tests for math ufuncs."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestSqrt:
    """Test sqrt ufunc."""

    def test_sqrt_1d(self):
        r = rp.sqrt(rp.arange(0, 10, dtype="float64"))
        n = np.sqrt(np.arange(10, dtype=np.float64))
        assert_eq(r, n)

    def test_sqrt_2d(self):
        r = rp.sqrt(rp.arange(1, 13, dtype="float64").reshape(3, 4))
        n = np.sqrt(np.arange(1, 13, dtype=np.float64).reshape(3, 4))
        assert_eq(r, n)


class TestExp:
    """Test exp ufunc."""

    def test_exp_1d(self):
        r = rp.exp(rp.arange(0, 5, dtype="float64"))
        n = np.exp(np.arange(5, dtype=np.float64))
        assert_eq(r, n)

    def test_exp_negative(self):
        r = rp.exp(rp.arange(-3, 3, dtype="float64"))
        n = np.exp(np.arange(-3, 3, dtype=np.float64))
        assert_eq(r, n)

    def test_exp_2d(self):
        r = rp.exp(rp.arange(6, dtype="float64").reshape(2, 3))
        n = np.exp(np.arange(6, dtype=np.float64).reshape(2, 3))
        assert_eq(r, n)


class TestLog:
    """Test log (natural logarithm) ufunc."""

    def test_log_1d(self):
        r = rp.log(rp.arange(1, 10, dtype="float64"))
        n = np.log(np.arange(1, 10, dtype=np.float64))
        assert_eq(r, n)

    def test_log_2d(self):
        r = rp.log(rp.arange(1, 7, dtype="float64").reshape(2, 3))
        n = np.log(np.arange(1, 7, dtype=np.float64).reshape(2, 3))
        assert_eq(r, n)

    def test_log_exp_inverse(self):
        x = rp.arange(1, 5, dtype="float64")
        assert_eq(rp.exp(rp.log(x)), np.arange(1, 5, dtype=np.float64))


class TestSin:
    """Test sin ufunc."""

    def test_sin_1d(self):
        r = rp.sin(rp.linspace(0, 6.28, 10))
        n = np.sin(np.linspace(0, 6.28, 10))
        assert_eq(r, n)

    def test_sin_2d(self):
        r = rp.sin(rp.linspace(0, 3.14, 6).reshape(2, 3))
        n = np.sin(np.linspace(0, 3.14, 6).reshape(2, 3))
        assert_eq(r, n)


class TestCos:
    """Test cos ufunc."""

    def test_cos_1d(self):
        r = rp.cos(rp.linspace(0, 6.28, 10))
        n = np.cos(np.linspace(0, 6.28, 10))
        assert_eq(r, n)

    def test_cos_2d(self):
        r = rp.cos(rp.linspace(0, 3.14, 6).reshape(2, 3))
        n = np.cos(np.linspace(0, 3.14, 6).reshape(2, 3))
        assert_eq(r, n)


class TestTan:
    """Test tan ufunc."""

    def test_tan_1d(self):
        # Avoid pi/2 where tan is undefined
        r = rp.tan(rp.linspace(0, 1, 10))
        n = np.tan(np.linspace(0, 1, 10))
        assert_eq(r, n)

    def test_tan_2d(self):
        r = rp.tan(rp.linspace(0, 1, 6).reshape(2, 3))
        n = np.tan(np.linspace(0, 1, 6).reshape(2, 3))
        assert_eq(r, n)


class TestSinCosPythagorean:
    """Test sin^2 + cos^2 = 1."""

    def test_pythagorean_identity(self):
        x = rp.linspace(0, 6.28, 100)
        s = rp.sin(x)
        c = rp.cos(x)
        result = s * s + c * c
        expected = rp.ones(100)
        assert_eq(result, np.asarray(expected))


class TestTranscendentalPromotion:
    """Test that transcendentals return same dtype and values as NumPy."""

    @pytest.mark.parametrize(
        "dtype",
        [
            "int64", "int32", "int16",
            "uint64", "uint32", "uint16", "uint8",
            "float64", "float32", "float16",
            "complex128",
        ],
    )
    @pytest.mark.parametrize("op", ["exp", "log", "sqrt", "sin", "cos", "tan"])
    def test_transcendental_dtype_and_values(self, dtype, op):
        """Transcendentals should return same dtype and values as NumPy."""
        rp_op = getattr(rp, op)
        np_op = getattr(np, op)

        r_arr = rp.asarray([1, 2, 4], dtype=dtype)
        n_arr = np.array([1, 2, 4], dtype=dtype)

        r_result = rp_op(r_arr)
        n_result = np_op(n_arr)

        # Verify dtypes match exactly
        assert r_result.dtype == str(n_result.dtype), f"Expected {n_result.dtype}, got {r_result.dtype}"
        # Compare values
        assert_eq(r_result, n_result)


ALL_DTYPES = [
    "int64", "int32", "int16",
    "uint64", "uint32", "uint16", "uint8",
    "float64", "float32", "float16",
    "complex128",
]

# Exclude complex128 for operations not yet implemented
NUMERIC_DTYPES = [d for d in ALL_DTYPES if d != "complex128"]


class TestBinaryDtypePromotion:
    """Test binary ops return same dtype and values as NumPy across dtype pairs."""

    @pytest.mark.parametrize("dtype_a", ALL_DTYPES)
    @pytest.mark.parametrize("dtype_b", ALL_DTYPES)
    def test_div_dtype_and_values(self, dtype_a, dtype_b):
        """Division should return same dtype and values as NumPy."""
        r_a = rp.asarray([6, 12, 24], dtype=dtype_a)
        r_b = rp.asarray([2, 3, 4], dtype=dtype_b)
        n_a = np.array([6, 12, 24], dtype=dtype_a)
        n_b = np.array([2, 3, 4], dtype=dtype_b)

        r_result = r_a / r_b
        n_result = n_a / n_b

        assert r_result.dtype == str(n_result.dtype), f"{dtype_a}/{dtype_b}: Expected {n_result.dtype}, got {r_result.dtype}"
        assert_eq(r_result, n_result)

    @pytest.mark.parametrize("dtype_a", ALL_DTYPES)
    @pytest.mark.parametrize("dtype_b", ALL_DTYPES)
    def test_add_dtype_and_values(self, dtype_a, dtype_b):
        """Addition should return same dtype and values as NumPy."""
        r_a = rp.asarray([1, 2, 3], dtype=dtype_a)
        r_b = rp.asarray([4, 5, 6], dtype=dtype_b)
        n_a = np.array([1, 2, 3], dtype=dtype_a)
        n_b = np.array([4, 5, 6], dtype=dtype_b)

        r_result = r_a + r_b
        n_result = n_a + n_b

        assert r_result.dtype == str(n_result.dtype), f"{dtype_a}+{dtype_b}: Expected {n_result.dtype}, got {r_result.dtype}"
        assert_eq(r_result, n_result)

    @pytest.mark.parametrize("dtype_a", NUMERIC_DTYPES)
    @pytest.mark.parametrize("dtype_b", NUMERIC_DTYPES)
    def test_pow_dtype_and_values(self, dtype_a, dtype_b):
        """Power should return same dtype and values as NumPy."""
        r_a = rp.asarray([2, 3, 4], dtype=dtype_a)
        r_b = rp.asarray([2, 2, 2], dtype=dtype_b)
        n_a = np.array([2, 3, 4], dtype=dtype_a)
        n_b = np.array([2, 2, 2], dtype=dtype_b)

        r_result = r_a ** r_b
        n_result = n_a ** n_b

        assert r_result.dtype == str(n_result.dtype), f"{dtype_a}**{dtype_b}: Expected {n_result.dtype}, got {r_result.dtype}"
        assert_eq(r_result, n_result)
