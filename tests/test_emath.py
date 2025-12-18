"""Tests for numpy.emath module - math with automatic complex domain extension.

These functions return complex results when real input would give NaN.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES, FLOAT_DTYPES
from helpers import assert_eq


def assert_scalar_eq(rp_result, np_result, rtol=1e-7):
    """Compare scalar results. rumpy returns arrays, numpy returns scalars."""
    rp_val = np.asarray(rp_result).flat[0]
    np.testing.assert_allclose(rp_val, np_result, rtol=rtol)


class TestEmathSqrt:
    """Test emath.sqrt: returns complex for negative input."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_positive_values(self, dtype):
        """Positive inputs return real results."""
        n = np.array([1, 4, 9, 16], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.sqrt(r), np.emath.sqrt(n))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative_values(self, dtype):
        """Negative inputs return complex results."""
        n = np.array([-1, -4, -9, -16], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.sqrt(r), np.emath.sqrt(n))

    def test_mixed_values(self):
        """Mixed positive/negative inputs."""
        n = np.array([-4, -1, 0, 1, 4], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.emath.sqrt(r), np.emath.sqrt(n))

    def test_zero(self):
        """sqrt(0) = 0."""
        n = np.array([0.0])
        r = rp.asarray(n)
        assert_eq(rp.emath.sqrt(r), np.emath.sqrt(n))

    def test_scalar(self):
        """Test scalar input (values match, shape may differ)."""
        assert_scalar_eq(rp.emath.sqrt(-1), np.emath.sqrt(-1))
        assert_scalar_eq(rp.emath.sqrt(4), np.emath.sqrt(4))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various shapes."""
        n = np.linspace(-4, 4, int(np.prod(shape))).reshape(shape)
        r = rp.asarray(n)
        assert_eq(rp.emath.sqrt(r), np.emath.sqrt(n))


class TestEmathLog:
    """Test emath.log: returns complex for negative input."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_positive_values(self, dtype):
        """Positive inputs return real results."""
        n = np.array([1, np.e, np.e**2], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.log(r), np.emath.log(n))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative_values(self, dtype):
        """Negative inputs return complex results."""
        n = np.array([-1, -np.e, -np.e**2], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.log(r), np.emath.log(n))

    def test_mixed_values(self):
        """Mixed positive/negative inputs."""
        n = np.array([-np.e, -1, 1, np.e], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.emath.log(r), np.emath.log(n))

    def test_scalar(self):
        """Test scalar input."""
        assert_scalar_eq(rp.emath.log(-1), np.emath.log(-1))
        assert_scalar_eq(rp.emath.log(np.e), np.emath.log(np.e))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various shapes."""
        # Avoid values too close to zero
        n = np.linspace(-4, 4, int(np.prod(shape))).reshape(shape)
        n = np.where(np.abs(n) < 0.5, np.sign(n) * 0.5, n)
        r = rp.asarray(n)
        assert_eq(rp.emath.log(r), np.emath.log(n))


class TestEmathLog2:
    """Test emath.log2: returns complex for negative input."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_positive_values(self, dtype):
        """Positive inputs return real results."""
        n = np.array([1, 2, 4, 8], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.log2(r), np.emath.log2(n))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative_values(self, dtype):
        """Negative inputs return complex results."""
        n = np.array([-1, -2, -4, -8], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.log2(r), np.emath.log2(n))

    def test_scalar(self):
        """Test scalar input."""
        assert_scalar_eq(rp.emath.log2(-1), np.emath.log2(-1))
        assert_scalar_eq(rp.emath.log2(2), np.emath.log2(2))


class TestEmathLog10:
    """Test emath.log10: returns complex for negative input."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_positive_values(self, dtype):
        """Positive inputs return real results."""
        n = np.array([1, 10, 100], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.log10(r), np.emath.log10(n))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative_values(self, dtype):
        """Negative inputs return complex results."""
        n = np.array([-1, -10, -100], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.log10(r), np.emath.log10(n))

    def test_scalar(self):
        """Test scalar input."""
        assert_scalar_eq(rp.emath.log10(-1), np.emath.log10(-1))
        assert_scalar_eq(rp.emath.log10(10), np.emath.log10(10))


class TestEmathLogn:
    """Test emath.logn: log with arbitrary base, complex domain."""

    def test_basic(self):
        """Test basic functionality."""
        # logn(base, x) - scalar inputs
        assert_scalar_eq(rp.emath.logn(10, 100), np.emath.logn(10, 100))
        assert_scalar_eq(rp.emath.logn(2, 8), np.emath.logn(2, 8))

    def test_negative_x(self):
        """Negative x returns complex."""
        assert_scalar_eq(rp.emath.logn(10, -100), np.emath.logn(10, -100))

    def test_negative_base(self):
        """Negative base returns complex."""
        assert_scalar_eq(rp.emath.logn(-2, 8), np.emath.logn(-2, 8))

    def test_arrays(self):
        """Test with array inputs."""
        base = np.array([2, 10], dtype=np.float64)
        x = np.array([8, 100], dtype=np.float64)
        r_base, r_x = rp.asarray(base), rp.asarray(x)
        assert_eq(rp.emath.logn(r_base, r_x), np.emath.logn(base, x))


class TestEmathPower:
    """Test emath.power: returns complex for negative base with fractional exponent."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_positive_base(self, dtype):
        """Positive base returns real."""
        x = np.array([1, 4, 9], dtype=dtype)
        p = np.array([0.5, 0.5, 0.5], dtype=dtype)
        rx, rp_ = rp.asarray(x), rp.asarray(p)
        assert_eq(rp.emath.power(rx, rp_), np.emath.power(x, p))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative_base_fractional(self, dtype):
        """Negative base with fractional exponent returns complex."""
        x = np.array([-1, -4, -9], dtype=dtype)
        p = np.array([0.5, 0.5, 0.5], dtype=dtype)
        rx, rp_ = rp.asarray(x), rp.asarray(p)
        assert_eq(rp.emath.power(rx, rp_), np.emath.power(x, p))

    def test_negative_base_integer(self):
        """Negative base with integer exponent works normally."""
        x = np.array([-1, -2, -3], dtype=np.float64)
        p = np.array([2, 2, 2], dtype=np.float64)
        rx, rp_ = rp.asarray(x), rp.asarray(p)
        assert_eq(rp.emath.power(rx, rp_), np.emath.power(x, p))

    def test_scalar(self):
        """Test scalar input."""
        assert_scalar_eq(rp.emath.power(-1, 0.5), np.emath.power(-1, 0.5))
        assert_scalar_eq(rp.emath.power(4, 0.5), np.emath.power(4, 0.5))


class TestEmathArccos:
    """Test emath.arccos: returns complex outside [-1, 1]."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_in_domain(self, dtype):
        """Values in [-1, 1] return real."""
        n = np.array([-1, -0.5, 0, 0.5, 1], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.arccos(r), np.emath.arccos(n))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_outside_domain(self, dtype):
        """Values outside [-1, 1] return complex."""
        n = np.array([-2, 2], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.arccos(r), np.emath.arccos(n))

    def test_mixed(self):
        """Test mixed in/out of domain."""
        n = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.emath.arccos(r), np.emath.arccos(n))

    def test_scalar(self):
        """Test scalar input."""
        assert_scalar_eq(rp.emath.arccos(2), np.emath.arccos(2))
        assert_scalar_eq(rp.emath.arccos(0.5), np.emath.arccos(0.5))


class TestEmathArcsin:
    """Test emath.arcsin: returns complex outside [-1, 1]."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_in_domain(self, dtype):
        """Values in [-1, 1] return real."""
        n = np.array([-1, -0.5, 0, 0.5, 1], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.arcsin(r), np.emath.arcsin(n))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_outside_domain(self, dtype):
        """Values outside [-1, 1] return complex."""
        n = np.array([-2, 2], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.arcsin(r), np.emath.arcsin(n))

    def test_mixed(self):
        """Test mixed in/out of domain."""
        n = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.emath.arcsin(r), np.emath.arcsin(n))

    def test_scalar(self):
        """Test scalar input."""
        assert_scalar_eq(rp.emath.arcsin(2), np.emath.arcsin(2))
        assert_scalar_eq(rp.emath.arcsin(0.5), np.emath.arcsin(0.5))


class TestEmathArctanh:
    """Test emath.arctanh: returns complex outside (-1, 1)."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_in_domain(self, dtype):
        """Values in (-1, 1) return real."""
        n = np.array([-0.9, -0.5, 0, 0.5, 0.9], dtype=dtype)
        r = rp.asarray(n)
        # arctanh has lower precision for float32
        rtol = 1e-5 if dtype == "float32" else 1e-7
        assert_eq(rp.emath.arctanh(r), np.emath.arctanh(n), rtol=rtol)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_outside_domain(self, dtype):
        """Values outside (-1, 1) return complex."""
        n = np.array([-2, 2], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.emath.arctanh(r), np.emath.arctanh(n))

    def test_mixed(self):
        """Test mixed in/out of domain."""
        n = np.array([-2, -0.5, 0, 0.5, 2], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.emath.arctanh(r), np.emath.arctanh(n))

    def test_scalar(self):
        """Test scalar input."""
        assert_scalar_eq(rp.emath.arctanh(2), np.emath.arctanh(2))
        assert_scalar_eq(rp.emath.arctanh(0.5), np.emath.arctanh(0.5))


class TestEmathDtypePreservation:
    """Test that emath preserves appropriate dtypes."""

    def test_sqrt_float32_positive(self):
        """float32 with all positive returns float32."""
        n = np.array([1, 4, 9], dtype=np.float32)
        r = rp.asarray(n)
        np_result = np.emath.sqrt(n)
        rp_result = rp.emath.sqrt(r)
        assert_eq(rp_result, np_result)
        # Check dtype matches
        assert str(rp_result.dtype) == str(np_result.dtype)

    def test_sqrt_float32_negative(self):
        """float32 with negative returns complex64."""
        n = np.array([-1, 4], dtype=np.float32)
        r = rp.asarray(n)
        np_result = np.emath.sqrt(n)
        rp_result = rp.emath.sqrt(r)
        assert_eq(rp_result, np_result)
        # Check dtype matches
        assert str(rp_result.dtype) == str(np_result.dtype)

    def test_sqrt_int_input(self):
        """int input with negative returns complex128."""
        n = np.array([-1, 4], dtype=np.int64)
        r = rp.asarray(n)
        np_result = np.emath.sqrt(n)
        rp_result = rp.emath.sqrt(r)
        assert_eq(rp_result, np_result)
