"""Tests for unary math operations (ufuncs).

Parametrizes over operations for concise comprehensive coverage.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES, FLOAT_DTYPES
from helpers import assert_eq, make_positive_pair

# === Ufunc categories by input domain ===

# Ufuncs that work on all real numbers
UNRESTRICTED_UFUNCS = ["exp", "sin", "cos", "tan", "sinh", "cosh", "tanh", "abs", "sign", "square", "negative", "positive"]

# Ufuncs that require positive input (x > 0)
POSITIVE_UFUNCS = ["sqrt", "log", "log10", "log2", "cbrt"]

# Ufuncs that require strictly positive input for safe testing
STRICTLY_POSITIVE_UFUNCS = ["reciprocal"]

# Ufuncs that work on [-1, 1]
BOUNDED_UFUNCS = ["arcsin", "arccos", "arctanh"]

# Ufuncs that work on [1, inf)
GE_ONE_UFUNCS = ["arccosh"]

# Rounding ufuncs
ROUNDING_UFUNCS = ["floor", "ceil", "trunc"]


# === Parametrized tests by category ===


class TestUnrestrictedUfuncs:
    """Test ufuncs that work on all real numbers."""

    @pytest.mark.parametrize("ufunc", UNRESTRICTED_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n = np.array([-2, -1, 0, 1, 2], dtype=dtype)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r), np_fn(n))

    @pytest.mark.parametrize("ufunc", UNRESTRICTED_UFUNCS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, ufunc, shape):
        n = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) - np.prod(shape) // 2
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r), np_fn(n))

    @pytest.mark.parametrize("ufunc", UNRESTRICTED_UFUNCS)
    def test_empty(self, ufunc):
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        result = rp_fn(r)
        expected = np_fn(n)
        assert result.shape == expected.shape


class TestPositiveUfuncs:
    """Test ufuncs that require positive input."""

    @pytest.mark.parametrize("ufunc", POSITIVE_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n = np.array([0.5, 1, 2, 4, 8], dtype=dtype)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r), np_fn(n))

    @pytest.mark.parametrize("ufunc", POSITIVE_UFUNCS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, ufunc, shape):
        r, n = make_positive_pair(shape, "float64")
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r), np_fn(n))


class TestStrictlyPositiveUfuncs:
    """Test ufuncs that require x > 0."""

    @pytest.mark.parametrize("ufunc", STRICTLY_POSITIVE_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n = np.array([0.5, 1, 2, 4], dtype=dtype)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r), np_fn(n))


class TestBoundedUfuncs:
    """Test ufuncs that require input in [-1, 1]."""

    @pytest.mark.parametrize("ufunc", BOUNDED_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n = np.array([-0.9, -0.5, 0, 0.5, 0.9], dtype=dtype)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        # float32 has lower precision
        rtol = 1e-5 if dtype == "float32" else 1e-7
        assert_eq(rp_fn(r), np_fn(n), rtol=rtol)


class TestGeOneUfuncs:
    """Test ufuncs that require input >= 1."""

    @pytest.mark.parametrize("ufunc", GE_ONE_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n = np.array([1, 2, 3, 4], dtype=dtype)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r), np_fn(n))


class TestRoundingUfuncs:
    """Test rounding ufuncs."""

    @pytest.mark.parametrize("ufunc", ROUNDING_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n = np.array([-1.7, -1.2, 0.3, 1.5, 2.9], dtype=dtype)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r), np_fn(n))

    @pytest.mark.parametrize("ufunc", ROUNDING_UFUNCS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, ufunc, shape):
        r, n = make_positive_pair(shape, "float64")
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r), np_fn(n))


# === Additional ufuncs with specific behavior ===


class TestArctan:
    """Test arctan (unrestricted domain)."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([-10, -1, 0, 1, 10], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.arctan(r), np.arctan(n))


class TestArcsinh:
    """Test arcsinh (unrestricted domain)."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([-2, -1, 0, 1, 2], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.arcsinh(r), np.arcsinh(n))


class TestExp2:
    """Test exp2 (2^x)."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([0, 1, 2, 3], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.exp2(r), np.exp2(n))


class TestExpm1Log1p:
    """Test expm1 and log1p for numerical stability near zero."""

    def test_expm1(self):
        n = np.array([0, 1e-10, 0.1, 1], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.expm1(r), np.expm1(n))

    def test_log1p(self):
        n = np.array([0, 1e-10, 0.1, 1], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.log1p(r), np.log1p(n))


class TestRintRound:
    """Test rint and round (banker's rounding)."""

    def test_rint(self):
        # Avoid -0.5 which has signed zero edge case
        n = np.array([-1.7, -0.4, 0.3, 1.5, 2.9], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.rint(r), np.rint(n))

    def test_round(self):
        n = np.array([-1.7, -0.4, 0.3, 1.5, 2.9], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.round(r), np.round(n))

    def test_round_decimals(self):
        n = np.array([1.234, 5.678, 9.012], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.round(r, decimals=2), np.round(n, decimals=2))


# === Testing predicates ===


class TestTestingFunctions:
    """Test isnan, isinf, isfinite, signbit."""

    def test_isnan(self):
        n = np.array([0, np.nan, 1, np.nan, 2], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.isnan(r), np.isnan(n))

    def test_isinf(self):
        n = np.array([0, np.inf, -np.inf, 1], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.isinf(r), np.isinf(n))

    def test_isfinite(self):
        n = np.array([0, np.inf, -np.inf, np.nan, 1], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.isfinite(r), np.isfinite(n))

    def test_signbit(self):
        n = np.array([-1.0, -0.0, 0.0, 1.0], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.signbit(r), np.signbit(n))


# === NaN handling ===


class TestNanToNum:
    """Test nan_to_num."""

    def test_basic(self):
        n = np.array([0, np.nan, np.inf, -np.inf, 1], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.nan_to_num(r), np.nan_to_num(n))

    def test_custom_values(self):
        n = np.array([0, np.nan, np.inf, -np.inf], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(
            rp.nan_to_num(r, nan=99, posinf=999, neginf=-999),
            np.nan_to_num(n, nan=99, posinf=999, neginf=-999),
        )


# === Complex number accessors ===


class TestComplexAccessors:
    """Test real, imag, conj for complex numbers."""

    def test_real_complex(self):
        n = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        r = rp.asarray(n)
        assert_eq(rp.real(r), np.real(n))

    def test_imag_complex(self):
        n = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        r = rp.asarray(n)
        assert_eq(rp.imag(r), np.imag(n))

    def test_conj_complex(self):
        n = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        r = rp.asarray(n)
        assert_eq(rp.conj(r), np.conj(n))

    def test_real_float(self):
        n = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.real(r), np.real(n))

    def test_imag_float(self):
        n = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.imag(r), np.imag(n))


# === Mathematical identities ===


class TestIdentities:
    """Test mathematical identities as sanity checks."""

    def test_sin_cos_pythagorean(self):
        n = np.linspace(0, 2 * np.pi, 100)
        r = rp.asarray(n)
        result = rp.sin(r) ** 2 + rp.cos(r) ** 2
        expected = np.ones(100)
        assert_eq(result, expected)

    def test_exp_log_inverse(self):
        n = np.array([1, 2, 3, 4], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.exp(rp.log(r)), n)

    def test_sinh_cosh_identity(self):
        """cosh^2(x) - sinh^2(x) = 1"""
        n = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        r = rp.asarray(n)
        result = rp.cosh(r) ** 2 - rp.sinh(r) ** 2
        expected = np.ones(5)
        assert_eq(result, expected)
