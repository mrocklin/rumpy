"""Tests for special mathematical functions.

Covers: sinc, i0, gcd, lcm, modf, frexp, ldexp, heaviside, spacing.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES, FLOAT_DTYPES, INT_DTYPES, UINT_DTYPES
from helpers import assert_eq


class TestSinc:
    """Test sinc function: sin(pi*x)/(pi*x)."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        # Avoid values where sinc -> 0 (integers) which have precision issues
        n = np.array([-0.7, -0.5, 0, 0.5, 0.7], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.sinc(r), np.sinc(n))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        n = np.linspace(-2, 2, int(np.prod(shape))).reshape(shape)
        r = rp.asarray(n)
        assert_eq(rp.sinc(r), np.sinc(n))

    def test_zero(self):
        """sinc(0) = 1."""
        n = np.array([0.0])
        r = rp.asarray(n)
        assert_eq(rp.sinc(r), np.sinc(n))

    def test_integers(self):
        """sinc(n) = 0 for non-zero integers."""
        n = np.array([-3, -2, -1, 1, 2, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.sinc(r), np.sinc(n))


class TestI0:
    """Test modified Bessel function of order 0."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([-1, -0.5, 0, 0.5, 1], dtype=dtype)
        r = rp.asarray(n)
        # float32 has slightly lower precision
        rtol = 1e-6 if dtype == "float32" else 1e-7
        assert_eq(rp.i0(r), np.i0(n), rtol=rtol)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        n = np.linspace(-3, 3, int(np.prod(shape))).reshape(shape)
        r = rp.asarray(n)
        assert_eq(rp.i0(r), np.i0(n))

    def test_zero(self):
        """i0(0) = 1."""
        n = np.array([0.0])
        r = rp.asarray(n)
        assert_eq(rp.i0(r), np.i0(n))

    def test_symmetry(self):
        """i0 is an even function: i0(-x) = i0(x)."""
        n = np.array([0.5, 1.0, 2.0, 3.0])
        r_pos = rp.asarray(n)
        r_neg = rp.asarray(-n)
        assert_eq(rp.i0(r_pos), rp.i0(r_neg))


class TestGcd:
    """Test greatest common divisor (element-wise)."""

    @pytest.mark.parametrize("dtype", INT_DTYPES + UINT_DTYPES)
    def test_dtypes(self, dtype):
        a = np.array([12, 15, 18, 24], dtype=dtype)
        b = np.array([8, 20, 24, 36], dtype=dtype)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.gcd(ra, rb), np.gcd(a, b))

    def test_shapes(self):
        a = np.array([[12, 15], [18, 24]])
        b = np.array([[8, 20], [24, 36]])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.gcd(ra, rb), np.gcd(a, b))

    def test_gcd_with_zero(self):
        """gcd(x, 0) = x."""
        a = np.array([5, 10, 15])
        b = np.array([0, 0, 0])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.gcd(ra, rb), np.gcd(a, b))

    def test_coprime(self):
        """Coprime numbers have gcd = 1."""
        a = np.array([7, 11, 13])
        b = np.array([3, 5, 9])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.gcd(ra, rb), np.gcd(a, b))


class TestLcm:
    """Test least common multiple (element-wise)."""

    @pytest.mark.parametrize("dtype", INT_DTYPES + UINT_DTYPES)
    def test_dtypes(self, dtype):
        a = np.array([12, 15, 18, 24], dtype=dtype)
        b = np.array([8, 20, 24, 36], dtype=dtype)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.lcm(ra, rb), np.lcm(a, b))

    def test_shapes(self):
        a = np.array([[12, 15], [18, 24]])
        b = np.array([[8, 20], [24, 36]])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.lcm(ra, rb), np.lcm(a, b))

    def test_lcm_with_zero(self):
        """lcm(x, 0) = 0."""
        a = np.array([5, 10, 15])
        b = np.array([0, 0, 0])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.lcm(ra, rb), np.lcm(a, b))


class TestModf:
    """Test modf: return fractional and integer parts."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([-1.5, -0.5, 0.5, 1.5, 2.7], dtype=dtype)
        r = rp.asarray(n)
        np_frac, np_int = np.modf(n)
        rp_frac, rp_int = rp.modf(r)
        assert_eq(rp_frac, np_frac)
        assert_eq(rp_int, np_int)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        n = np.linspace(-3.5, 3.5, int(np.prod(shape))).reshape(shape)
        r = rp.asarray(n)
        np_frac, np_int = np.modf(n)
        rp_frac, rp_int = rp.modf(r)
        assert_eq(rp_frac, np_frac)
        assert_eq(rp_int, np_int)

    def test_negative(self):
        """Check sign handling for negative numbers."""
        n = np.array([-2.7])
        r = rp.asarray(n)
        np_frac, np_int = np.modf(n)
        rp_frac, rp_int = rp.modf(r)
        assert_eq(rp_frac, np_frac)
        assert_eq(rp_int, np_int)


class TestFrexp:
    """Test frexp: decompose into mantissa and exponent."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=dtype)
        r = rp.asarray(n)
        np_mant, np_exp = np.frexp(n)
        rp_mant, rp_exp = rp.frexp(r)
        assert_eq(rp_mant, np_mant)
        assert_eq(rp_exp, np_exp)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        n = np.abs(np.linspace(0.1, 10, int(np.prod(shape)))).reshape(shape)
        r = rp.asarray(n)
        np_mant, np_exp = np.frexp(n)
        rp_mant, rp_exp = rp.frexp(r)
        assert_eq(rp_mant, np_mant)
        assert_eq(rp_exp, np_exp)

    def test_zero(self):
        """frexp(0) = (0, 0)."""
        n = np.array([0.0])
        r = rp.asarray(n)
        np_mant, np_exp = np.frexp(n)
        rp_mant, rp_exp = rp.frexp(r)
        assert_eq(rp_mant, np_mant)
        assert_eq(rp_exp, np_exp)


class TestLdexp:
    """Test ldexp: x * 2^i."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        x = np.array([0.5, 0.5, 0.5, 0.5], dtype=dtype)
        i = np.array([0, 1, 2, 3], dtype=np.int32)
        rx, ri = rp.asarray(x), rp.asarray(i)
        assert_eq(rp.ldexp(rx, ri), np.ldexp(x, i))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        size = int(np.prod(shape))
        x = np.ones(size, dtype=np.float64).reshape(shape)
        i = np.arange(size, dtype=np.int32).reshape(shape)
        rx, ri = rp.asarray(x), rp.asarray(i)
        assert_eq(rp.ldexp(rx, ri), np.ldexp(x, i))

    def test_negative_exponent(self):
        x = np.array([1.0, 2.0, 4.0])
        i = np.array([-1, -2, -3], dtype=np.int32)
        rx, ri = rp.asarray(x), rp.asarray(i)
        assert_eq(rp.ldexp(rx, ri), np.ldexp(x, i))


class TestHeaviside:
    """Test Heaviside step function."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        x = np.array([-1.5, -0.5, 0, 0.5, 1.5], dtype=dtype)
        h0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype)
        rx, rh0 = rp.asarray(x), rp.asarray(h0)
        assert_eq(rp.heaviside(rx, rh0), np.heaviside(x, h0))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        size = int(np.prod(shape))
        x = np.linspace(-2, 2, size).reshape(shape)
        h0 = np.full(shape, 0.5)
        rx, rh0 = rp.asarray(x), rp.asarray(h0)
        assert_eq(rp.heaviside(rx, rh0), np.heaviside(x, h0))

    def test_h0_values(self):
        """Test different h0 values at zero."""
        x = np.array([0.0, 0.0, 0.0])
        h0 = np.array([0.0, 0.5, 1.0])
        rx, rh0 = rp.asarray(x), rp.asarray(h0)
        assert_eq(rp.heaviside(rx, rh0), np.heaviside(x, h0))


class TestSpacing:
    """Test spacing: ULP distance."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([1.0, 1e10, 1e-10], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.spacing(r), np.spacing(n))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        n = np.abs(np.linspace(0.1, 100, int(np.prod(shape)))).reshape(shape)
        r = rp.asarray(n)
        assert_eq(rp.spacing(r), np.spacing(n))

    def test_special_values(self):
        """Test with edge values."""
        n = np.array([0.0, 1.0, np.finfo(np.float64).max / 2])
        r = rp.asarray(n)
        assert_eq(rp.spacing(r), np.spacing(n))
