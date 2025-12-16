"""Tests for convenience aliases and constants (Stream 30)."""

import numpy as np
import pytest

import rumpy as rp

from helpers import assert_eq


class TestConstants:
    """Test mathematical constants."""

    def test_pi(self):
        assert rp.pi == np.pi

    def test_e(self):
        assert rp.e == np.e

    def test_inf(self):
        assert rp.inf == np.inf
        assert np.isinf(rp.inf)

    def test_nan(self):
        assert np.isnan(rp.nan)

    def test_newaxis(self):
        assert rp.newaxis is None
        # Test usage in indexing
        n = np.arange(5)
        r = rp.arange(5)
        assert_eq(r[np.newaxis, :], n[np.newaxis, :])


class TestMathAliases:
    """Test math function aliases."""

    def test_absolute(self):
        n = np.array([-1.0, 2.0, -3.0])
        r = rp.asarray(n)
        assert_eq(rp.absolute(r), np.absolute(n))
        # Scalar
        assert rp.absolute(-5.0) == np.absolute(-5.0)

    def test_conjugate(self):
        n = np.array([1+2j, 3-4j])
        r = rp.asarray(n)
        assert_eq(rp.conjugate(r), np.conjugate(n))

    def test_asin(self):
        n = np.array([0.0, 0.5, 1.0])
        r = rp.asarray(n)
        assert_eq(rp.asin(r), np.asin(n))
        # Verify alias relationship
        assert_eq(rp.asin(r), rp.arcsin(r))

    def test_acos(self):
        n = np.array([0.0, 0.5, 1.0])
        r = rp.asarray(n)
        assert_eq(rp.acos(r), np.acos(n))
        assert_eq(rp.acos(r), rp.arccos(r))

    def test_atan(self):
        n = np.array([0.0, 1.0, -1.0])
        r = rp.asarray(n)
        assert_eq(rp.atan(r), np.atan(n))
        assert_eq(rp.atan(r), rp.arctan(r))

    def test_asinh(self):
        n = np.array([0.0, 1.0, -1.0])
        r = rp.asarray(n)
        assert_eq(rp.asinh(r), np.asinh(n))
        assert_eq(rp.asinh(r), rp.arcsinh(r))

    def test_acosh(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rp.asarray(n)
        assert_eq(rp.acosh(r), np.acosh(n))
        assert_eq(rp.acosh(r), rp.arccosh(r))

    def test_atanh(self):
        n = np.array([0.0, 0.5, -0.5])
        r = rp.asarray(n)
        assert_eq(rp.atanh(r), np.atanh(n))
        assert_eq(rp.atanh(r), rp.arctanh(r))

    def test_pow(self):
        n1 = np.array([2.0, 3.0, 4.0])
        n2 = np.array([1.0, 2.0, 3.0])
        r1 = rp.asarray(n1)
        r2 = rp.asarray(n2)
        assert_eq(rp.pow(r1, r2), np.power(n1, n2))
        assert_eq(rp.pow(r1, r2), rp.power(r1, r2))

    def test_mod(self):
        n1 = np.array([7.0, 8.0, 9.0])
        n2 = np.array([3.0, 3.0, 3.0])
        r1 = rp.asarray(n1)
        r2 = rp.asarray(n2)
        # Our mod uses C-style %, matches fmod not numpy's mod
        assert_eq(rp.mod(r1, r2), rp.remainder(r1, r2))

    def test_fmod(self):
        # fmod uses C-style modulo (sign of dividend preserved)
        n1 = np.array([7.0, -7.0, 7.0])
        n2 = np.array([3.0, 3.0, -3.0])
        r1 = rp.asarray(n1)
        r2 = rp.asarray(n2)
        assert_eq(rp.fmod(r1, r2), np.fmod(n1, n2))

    def test_true_divide(self):
        n1 = np.array([5.0, 6.0, 7.0])
        n2 = np.array([2.0, 2.0, 2.0])
        r1 = rp.asarray(n1)
        r2 = rp.asarray(n2)
        assert_eq(rp.true_divide(r1, r2), np.true_divide(n1, n2))
        assert_eq(rp.true_divide(r1, r2), rp.divide(r1, r2))

    def test_fabs(self):
        n = np.array([-1.5, 2.5, -3.5])
        r = rp.asarray(n)
        assert_eq(rp.fabs(r), np.fabs(n))
        assert_eq(rp.fabs(r), rp.abs(r))


class TestReductionAliases:
    """Test reduction function aliases."""

    def test_amax(self):
        n = np.array([[1, 2, 3], [4, 5, 6]])
        r = rp.asarray(n)
        # Global max
        assert rp.amax(r) == np.amax(n)
        # Axis max
        assert_eq(rp.amax(r, axis=0), np.amax(n, axis=0))
        assert_eq(rp.amax(r, axis=1), np.amax(n, axis=1))
        # Verify alias
        assert rp.amax(r) == rp.max(r)

    def test_amin(self):
        n = np.array([[1, 2, 3], [4, 5, 6]])
        r = rp.asarray(n)
        # Global min
        assert rp.amin(r) == np.amin(n)
        # Axis min
        assert_eq(rp.amin(r, axis=0), np.amin(n, axis=0))
        assert_eq(rp.amin(r, axis=1), np.amin(n, axis=1))
        # Verify alias
        assert rp.amin(r) == rp.min(r)


class TestScalarInputs:
    """Test aliases work with scalar inputs where applicable."""

    def test_absolute_scalar(self):
        assert rp.absolute(-3.14) == np.absolute(-3.14)

    def test_fabs_scalar(self):
        assert rp.fabs(-2.5) == np.fabs(-2.5)

    def test_asin_scalar(self):
        assert abs(rp.asin(0.5) - np.asin(0.5)) < 1e-10

    def test_pow_scalar(self):
        result = rp.pow(2.0, 3.0)
        # May return array or scalar depending on impl
        if hasattr(result, '__float__'):
            assert float(result) == 8.0
        else:
            assert result[0] == 8.0
