"""Tests for binary math operations.

Parametrizes over operations for concise comprehensive coverage.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES, FLOAT_DTYPES, NUMERIC_DTYPES, SHAPES_BROADCAST
from helpers import assert_eq, make_pair, make_positive_pair

# === Binary ufunc categories ===

# Arithmetic ops that work on all numbers
ARITHMETIC_UFUNCS = ["add", "subtract", "multiply"]

# Division (avoid zero)
DIVISION_UFUNCS = ["divide", "floor_divide"]

# Ops that need positive second operand
POWER_UFUNCS = ["power"]

# Comparison ops
COMPARISON_UFUNCS = ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]

# Min/max ops
MINMAX_UFUNCS = ["maximum", "minimum"]

# Two-argument trig
TRIG2_UFUNCS = ["arctan2", "hypot"]

# Float min/max with NaN handling
FMINMAX_UFUNCS = ["fmax", "fmin"]


# === Parametrized tests ===


class TestArithmeticUfuncs:
    """Test basic arithmetic: add, subtract, multiply."""

    @pytest.mark.parametrize("ufunc", ARITHMETIC_UFUNCS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n1 = np.array([1, 2, 3, 4, 5], dtype=dtype)
        n2 = np.array([5, 4, 3, 2, 1], dtype=dtype)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("ufunc", ARITHMETIC_UFUNCS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, ufunc, shape):
        r1, n1 = make_pair(shape, "float64")
        r2, n2 = make_pair(shape, "float64")
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("ufunc", ARITHMETIC_UFUNCS)
    def test_scalar(self, ufunc):
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r, 2.0), np_fn(n, 2.0))
        assert_eq(rp_fn(2.0, r), np_fn(2.0, n))

    @pytest.mark.parametrize("ufunc", ARITHMETIC_UFUNCS)
    @pytest.mark.parametrize("shapes", SHAPES_BROADCAST)
    def test_broadcast(self, ufunc, shapes):
        shape1, shape2 = shapes
        n1 = np.ones(shape1, dtype=np.float64)
        n2 = np.ones(shape2, dtype=np.float64) * 2
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))


class TestDivisionUfuncs:
    """Test division operations (avoid zero)."""

    @pytest.mark.parametrize("ufunc", DIVISION_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n1 = np.array([10, 20, 30, 40], dtype=dtype)
        n2 = np.array([2, 4, 5, 8], dtype=dtype)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("ufunc", DIVISION_UFUNCS)
    def test_scalar(self, ufunc):
        n = np.array([10, 20, 30, 40], dtype=np.float64)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r, 2.0), np_fn(n, 2.0))


class TestPowerUfuncs:
    """Test power operations."""

    @pytest.mark.parametrize("ufunc", POWER_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n1 = np.array([1, 2, 3, 4], dtype=dtype)
        n2 = np.array([2, 2, 2, 2], dtype=dtype)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("ufunc", POWER_UFUNCS)
    def test_scalar(self, ufunc):
        n = np.array([1, 2, 3, 4], dtype=np.float64)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r, 2), np_fn(n, 2))


class TestComparisonUfuncs:
    """Test comparison operations."""

    @pytest.mark.parametrize("ufunc", COMPARISON_UFUNCS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n1 = np.array([1, 2, 3, 4, 5], dtype=dtype)
        n2 = np.array([3, 3, 3, 3, 3], dtype=dtype)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("ufunc", COMPARISON_UFUNCS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, ufunc, shape):
        r1, n1 = make_pair(shape, "float64")
        r2, n2 = make_pair(shape, "float64")
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("ufunc", COMPARISON_UFUNCS)
    def test_scalar(self, ufunc):
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r, 3.0), np_fn(n, 3.0))


class TestMinMaxUfuncs:
    """Test element-wise maximum and minimum."""

    @pytest.mark.parametrize("ufunc", MINMAX_UFUNCS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n1 = np.array([1, 5, 2, 8, 3], dtype=dtype)
        n2 = np.array([4, 2, 6, 1, 9], dtype=dtype)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("ufunc", MINMAX_UFUNCS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, ufunc, shape):
        r1, n1 = make_pair(shape, "float64")
        r2, n2 = make_pair(shape, "float64")
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))


class TestTrig2Ufuncs:
    """Test two-argument trig: arctan2, hypot."""

    @pytest.mark.parametrize("ufunc", TRIG2_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n1 = np.array([1, 2, 3, 4], dtype=dtype)
        n2 = np.array([4, 3, 2, 1], dtype=dtype)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))


class TestFminFmax:
    """Test fmin/fmax (NaN-propagating min/max)."""

    @pytest.mark.parametrize("ufunc", FMINMAX_UFUNCS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, ufunc, dtype):
        n1 = np.array([1, 5, 2, 8], dtype=dtype)
        n2 = np.array([4, 2, 6, 1], dtype=dtype)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("ufunc", FMINMAX_UFUNCS)
    def test_with_nan(self, ufunc):
        n1 = np.array([1, np.nan, 3], dtype=np.float64)
        n2 = np.array([2, 2, np.nan], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, ufunc)
        np_fn = getattr(np, ufunc)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))


# === Modulo and remainder ===


class TestModRemainder:
    """Test modulo and remainder operations."""

    def test_remainder(self):
        n1 = np.array([7, 8, 9, 10], dtype=np.float64)
        n2 = np.array([3, 3, 3, 3], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.remainder(r1, r2), np.remainder(n1, n2))


# === Operator overloading (dunder methods) ===


class TestDunderOps:
    """Test operator overloading matches numpy."""

    def test_add_op(self):
        n1 = np.array([1, 2, 3], dtype=np.float64)
        n2 = np.array([4, 5, 6], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 + r2, n1 + n2)

    def test_sub_op(self):
        n1 = np.array([4, 5, 6], dtype=np.float64)
        n2 = np.array([1, 2, 3], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 - r2, n1 - n2)

    def test_mul_op(self):
        n1 = np.array([1, 2, 3], dtype=np.float64)
        n2 = np.array([4, 5, 6], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 * r2, n1 * n2)

    def test_truediv_op(self):
        n1 = np.array([10, 20, 30], dtype=np.float64)
        n2 = np.array([2, 4, 5], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 / r2, n1 / n2)

    def test_floordiv_op(self):
        n1 = np.array([10, 21, 35], dtype=np.float64)
        n2 = np.array([3, 4, 6], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 // r2, n1 // n2)

    def test_pow_op(self):
        n1 = np.array([2, 3, 4], dtype=np.float64)
        n2 = np.array([2, 2, 2], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 ** r2, n1 ** n2)

    def test_mod_op(self):
        n1 = np.array([7, 8, 9], dtype=np.float64)
        n2 = np.array([3, 3, 3], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 % r2, n1 % n2)

    def test_neg_op(self):
        n = np.array([1, -2, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(-r, -n)



class TestComparisonOps:
    """Test comparison operators."""

    def test_eq_op(self):
        n1 = np.array([1, 2, 3], dtype=np.float64)
        n2 = np.array([1, 3, 3], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 == r2, n1 == n2)

    def test_ne_op(self):
        n1 = np.array([1, 2, 3], dtype=np.float64)
        n2 = np.array([1, 3, 3], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 != r2, n1 != n2)

    def test_lt_op(self):
        n1 = np.array([1, 2, 3], dtype=np.float64)
        n2 = np.array([2, 2, 2], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 < r2, n1 < n2)

    def test_le_op(self):
        n1 = np.array([1, 2, 3], dtype=np.float64)
        n2 = np.array([2, 2, 2], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 <= r2, n1 <= n2)

    def test_gt_op(self):
        n1 = np.array([1, 2, 3], dtype=np.float64)
        n2 = np.array([2, 2, 2], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 > r2, n1 > n2)

    def test_ge_op(self):
        n1 = np.array([1, 2, 3], dtype=np.float64)
        n2 = np.array([2, 2, 2], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(r1 >= r2, n1 >= n2)


# === Clip and where ===


class TestClip:
    """Test clip operation."""

    def test_basic(self):
        n = np.array([1, 5, 10, 15, 20], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.clip(r, 5, 15), np.clip(n, 5, 15))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([1, 5, 10, 15, 20], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.clip(r, 5, 15), np.clip(n, 5, 15))


class TestWhere:
    """Test where operation."""

    def test_basic(self):
        cond = np.array([True, False, True, False, True])
        n1 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        n2 = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        r_cond = rp.asarray(cond)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.where(r_cond, r1, r2), np.where(cond, n1, n2))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        size = int(np.prod(shape))
        cond = np.array([i % 2 == 0 for i in range(size)]).reshape(shape)
        n1 = np.arange(size, dtype=np.float64).reshape(shape)
        n2 = np.arange(size, dtype=np.float64).reshape(shape) * 10
        r_cond = rp.asarray(cond)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.where(r_cond, r1, r2), np.where(cond, n1, n2))


# === isclose ===


class TestIsclose:
    """Test isclose for approximate equality."""

    def test_basic(self):
        n1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        n2 = np.array([1.0, 2.0 + 1e-10, 3.1], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.isclose(r1, r2), np.isclose(n1, n2))

    def test_with_tolerance(self):
        n1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        n2 = np.array([1.01, 2.01, 3.01], dtype=np.float64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(
            rp.isclose(r1, r2, rtol=0.1),
            np.isclose(n1, n2, rtol=0.1),
        )
