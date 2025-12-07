"""Tests for comparison operations."""
import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestEqual:
    def test_basic(self):
        a = rp.array([1, 2, 3, 4])
        b = rp.array([1, 0, 3, 0])
        r = rp.equal(a, b)
        n = np.equal([1, 2, 3, 4], [1, 0, 3, 0])
        assert_eq(r, n)

    def test_float(self):
        a = rp.array([1.0, 2.0, 3.0])
        b = rp.array([1.0, 2.1, 3.0])
        r = rp.equal(a, b)
        n = np.equal([1.0, 2.0, 3.0], [1.0, 2.1, 3.0])
        assert_eq(r, n)

    def test_broadcasting(self):
        a = rp.array([[1, 2], [3, 4]])
        b = rp.array([1, 2])
        r = rp.equal(a, b)
        n = np.equal([[1, 2], [3, 4]], [1, 2])
        assert_eq(r, n)

    def test_scalar_right(self):
        a = rp.array([1, 2, 3, 4])
        r = rp.equal(a, 2)
        n = np.equal([1, 2, 3, 4], 2)
        assert_eq(r, n)

    def test_scalar_left(self):
        b = rp.array([1, 2, 3, 4])
        r = rp.equal(2, b)
        n = np.equal(2, [1, 2, 3, 4])
        assert_eq(r, n)


class TestNotEqual:
    def test_basic(self):
        a = rp.array([1, 2, 3, 4])
        b = rp.array([1, 0, 3, 0])
        r = rp.not_equal(a, b)
        n = np.not_equal([1, 2, 3, 4], [1, 0, 3, 0])
        assert_eq(r, n)


class TestLess:
    def test_basic(self):
        a = rp.array([1, 2, 3, 4])
        b = rp.array([2, 2, 2, 2])
        r = rp.less(a, b)
        n = np.less([1, 2, 3, 4], [2, 2, 2, 2])
        assert_eq(r, n)

    def test_float(self):
        a = rp.array([1.0, 2.0, 3.0])
        b = rp.array([1.5, 1.5, 1.5])
        r = rp.less(a, b)
        n = np.less([1.0, 2.0, 3.0], [1.5, 1.5, 1.5])
        assert_eq(r, n)

    def test_scalar_right(self):
        a = rp.array([1, 2, 3, 4])
        r = rp.less(a, 3)
        n = np.less([1, 2, 3, 4], 3)
        assert_eq(r, n)

    def test_scalar_left(self):
        b = rp.array([1, 2, 3, 4])
        r = rp.less(1, b)
        n = np.less(1, [1, 2, 3, 4])
        assert_eq(r, n)


class TestLessEqual:
    def test_basic(self):
        a = rp.array([1, 2, 3, 4])
        b = rp.array([2, 2, 2, 2])
        r = rp.less_equal(a, b)
        n = np.less_equal([1, 2, 3, 4], [2, 2, 2, 2])
        assert_eq(r, n)


class TestGreater:
    def test_basic(self):
        a = rp.array([1, 2, 3, 4])
        b = rp.array([2, 2, 2, 2])
        r = rp.greater(a, b)
        n = np.greater([1, 2, 3, 4], [2, 2, 2, 2])
        assert_eq(r, n)


class TestGreaterEqual:
    def test_basic(self):
        a = rp.array([1, 2, 3, 4])
        b = rp.array([2, 2, 2, 2])
        r = rp.greater_equal(a, b)
        n = np.greater_equal([1, 2, 3, 4], [2, 2, 2, 2])
        assert_eq(r, n)


class TestIsclose:
    def test_exact(self):
        a = rp.array([1.0, 2.0, 3.0])
        b = rp.array([1.0, 2.0, 3.0])
        r = rp.isclose(a, b)
        n = np.isclose([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert_eq(r, n)

    def test_close(self):
        a = rp.array([1.0, 2.0, 3.0])
        b = rp.array([1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])
        r = rp.isclose(a, b)
        n = np.isclose([1.0, 2.0, 3.0], [1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])
        assert_eq(r, n)

    def test_not_close(self):
        a = rp.array([1.0, 2.0, 3.0])
        b = rp.array([1.1, 2.1, 3.1])
        r = rp.isclose(a, b)
        n = np.isclose([1.0, 2.0, 3.0], [1.1, 2.1, 3.1])
        assert_eq(r, n)

    def test_custom_tolerance(self):
        a = rp.array([1.0, 2.0])
        b = rp.array([1.05, 2.05])
        r = rp.isclose(a, b, rtol=0.1, atol=0.0)
        n = np.isclose([1.0, 2.0], [1.05, 2.05], rtol=0.1, atol=0.0)
        assert_eq(r, n)

    def test_nan_default(self):
        """NaN should not be close to NaN by default."""
        a = rp.array([1.0, float("nan"), 3.0])
        b = rp.array([1.0, float("nan"), 3.0])
        r = rp.isclose(a, b)
        n = np.isclose([1.0, np.nan, 3.0], [1.0, np.nan, 3.0])
        assert_eq(r, n)

    def test_nan_equal_nan(self):
        """With equal_nan=True, NaN should be close to NaN."""
        a = rp.array([1.0, float("nan"), 3.0])
        b = rp.array([1.0, float("nan"), 3.0])
        r = rp.isclose(a, b, equal_nan=True)
        n = np.isclose([1.0, np.nan, 3.0], [1.0, np.nan, 3.0], equal_nan=True)
        assert_eq(r, n)

    def test_inf(self):
        """Infinity should only be close to same-signed infinity."""
        a = rp.array([float("inf"), float("-inf"), 1.0, float("inf")])
        b = rp.array([float("inf"), float("-inf"), 1.0, float("-inf")])
        r = rp.isclose(a, b)
        n = np.isclose([np.inf, -np.inf, 1.0, np.inf], [np.inf, -np.inf, 1.0, -np.inf])
        assert_eq(r, n)


class TestAllclose:
    def test_true(self):
        a = rp.array([1.0, 2.0, 3.0])
        b = rp.array([1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])
        assert rp.allclose(a, b) == np.allclose([1.0, 2.0, 3.0], [1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])

    def test_false(self):
        a = rp.array([1.0, 2.0, 3.0])
        b = rp.array([1.1, 2.0, 3.0])
        assert rp.allclose(a, b) == np.allclose([1.0, 2.0, 3.0], [1.1, 2.0, 3.0])

    def test_nan_default(self):
        """Arrays with NaN should not be allclose by default."""
        a = rp.array([1.0, float("nan"), 3.0])
        b = rp.array([1.0, float("nan"), 3.0])
        assert rp.allclose(a, b) == np.allclose([1.0, np.nan, 3.0], [1.0, np.nan, 3.0])

    def test_nan_equal_nan(self):
        """With equal_nan=True, arrays with matching NaN should be allclose."""
        a = rp.array([1.0, float("nan"), 3.0])
        b = rp.array([1.0, float("nan"), 3.0])
        assert rp.allclose(a, b, equal_nan=True) == np.allclose([1.0, np.nan, 3.0], [1.0, np.nan, 3.0], equal_nan=True)


class TestArrayEqual:
    def test_equal(self):
        a = rp.array([1, 2, 3])
        b = rp.array([1, 2, 3])
        assert rp.array_equal(a, b) == np.array_equal([1, 2, 3], [1, 2, 3])

    def test_not_equal_values(self):
        a = rp.array([1, 2, 3])
        b = rp.array([1, 2, 4])
        assert rp.array_equal(a, b) == np.array_equal([1, 2, 3], [1, 2, 4])

    def test_not_equal_shape(self):
        a = rp.array([1, 2, 3])
        b = rp.array([1, 2])
        assert rp.array_equal(a, b) == np.array_equal([1, 2, 3], [1, 2])

    def test_2d(self):
        a = rp.array([[1, 2], [3, 4]])
        b = rp.array([[1, 2], [3, 4]])
        assert rp.array_equal(a, b) == np.array_equal([[1, 2], [3, 4]], [[1, 2], [3, 4]])


class TestOutputDtype:
    def test_equal_dtype(self):
        r = rp.equal(rp.array([1, 2]), rp.array([1, 3]))
        assert r.dtype == "bool"

    def test_less_dtype(self):
        r = rp.less(rp.array([1.0, 2.0]), rp.array([1.5, 1.5]))
        assert r.dtype == "bool"

    def test_isclose_dtype(self):
        r = rp.isclose(rp.array([1.0]), rp.array([1.0]))
        assert r.dtype == "bool"
