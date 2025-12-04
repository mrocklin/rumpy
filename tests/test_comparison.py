"""Tests for comparison operations."""

import numpy as np
import rumpy

from helpers import assert_eq


class TestComparisonScalar:
    """Test array vs scalar comparisons."""

    def test_gt_scalar(self):
        r = rumpy.arange(5) > 2
        n = np.arange(5) > 2
        assert_eq(r, n)

    def test_lt_scalar(self):
        r = rumpy.arange(5) < 2
        n = np.arange(5) < 2
        assert_eq(r, n)

    def test_ge_scalar(self):
        r = rumpy.arange(5) >= 2
        n = np.arange(5) >= 2
        assert_eq(r, n)

    def test_le_scalar(self):
        r = rumpy.arange(5) <= 2
        n = np.arange(5) <= 2
        assert_eq(r, n)

    def test_eq_scalar(self):
        r = rumpy.arange(5) == 2
        n = np.arange(5) == 2
        assert_eq(r, n)

    def test_ne_scalar(self):
        r = rumpy.arange(5) != 2
        n = np.arange(5) != 2
        assert_eq(r, n)


class TestComparisonArray:
    """Test array vs array comparisons."""

    def test_gt_array(self):
        a = rumpy.arange(5)
        b = rumpy.full([5], 2.0)
        r = a > b
        n = np.arange(5) > np.full(5, 2.0)
        assert_eq(r, n)

    def test_lt_array(self):
        a = rumpy.arange(5)
        b = rumpy.full([5], 2.0)
        r = a < b
        n = np.arange(5) < np.full(5, 2.0)
        assert_eq(r, n)

    def test_eq_array(self):
        a = rumpy.arange(5)
        b = rumpy.arange(5)
        r = a == b
        n = np.arange(5) == np.arange(5)
        assert_eq(r, n)

    def test_ne_array(self):
        a = rumpy.arange(5)
        b = rumpy.full([5], 2.0)
        r = a != b
        n = np.arange(5) != np.full(5, 2.0)
        assert_eq(r, n)


class TestComparisonBroadcast:
    """Test comparisons with broadcasting."""

    def test_gt_broadcast_1d_2d(self):
        a = rumpy.arange(3).reshape([1, 3])
        b = rumpy.arange(3).reshape([3, 1])
        r = a > b
        n = np.arange(3).reshape(1, 3) > np.arange(3).reshape(3, 1)
        assert_eq(r, n)

    def test_lt_broadcast_scalar_shape(self):
        a = rumpy.arange(6).reshape([2, 3])
        b = rumpy.full([1], 3.0)
        r = a < b
        n = np.arange(6).reshape(2, 3) < np.full([1], 3.0)
        assert_eq(r, n)


class TestComparisonDtype:
    """Test that comparison results have bool dtype."""

    def test_result_dtype_is_bool(self):
        r = rumpy.arange(5) > 2
        assert r.dtype == "bool"

    def test_result_dtype_array_comparison(self):
        a = rumpy.arange(5)
        b = rumpy.full([5], 2.0)
        r = a > b
        assert r.dtype == "bool"


class TestBooleanIndexing:
    """Test boolean array indexing (arr[mask])."""

    def test_bool_index_1d(self):
        arr = rumpy.arange(10)
        mask = arr > 5
        r = arr[mask]
        n = np.arange(10)[np.arange(10) > 5]
        assert_eq(r, n)

    def test_bool_index_returns_1d(self):
        arr = rumpy.arange(6).reshape([2, 3])
        mask = arr > 2
        r = arr[mask]
        n = np.arange(6).reshape(2, 3)[np.arange(6).reshape(2, 3) > 2]
        assert_eq(r, n)

    def test_bool_index_all_true(self):
        arr = rumpy.arange(5)
        mask = arr >= 0
        r = arr[mask]
        n = np.arange(5)[np.arange(5) >= 0]
        assert_eq(r, n)

    def test_bool_index_all_false(self):
        arr = rumpy.arange(5)
        mask = arr > 100
        r = arr[mask]
        n = np.arange(5)[np.arange(5) > 100]
        assert_eq(r, n)
        assert r.size == 0

    def test_bool_index_preserves_dtype(self):
        arr = rumpy.arange(5, dtype="int32")
        mask = arr > 2
        r = arr[mask]
        assert r.dtype == "int32"

    def test_bool_index_chained(self):
        arr = rumpy.arange(10)
        r = arr[arr > 5]
        n = np.arange(10)[np.arange(10) > 5]
        assert_eq(r, n)


class TestWhere:
    """Test where() conditional selection."""

    def test_where_scalar_values(self):
        cond = rumpy.arange(5) > 2
        r = rumpy.where(cond, 1, 0)
        n = np.where(np.arange(5) > 2, 1, 0)
        assert_eq(r, n)

    def test_where_array_values(self):
        cond = rumpy.arange(5) > 2
        x = rumpy.arange(5)
        y = rumpy.zeros([5])
        r = rumpy.where(cond, x, y)
        n = np.where(np.arange(5) > 2, np.arange(5), np.zeros(5))
        assert_eq(r, n)

    def test_where_mixed_scalar_array(self):
        cond = rumpy.arange(5) > 2
        x = rumpy.arange(5)
        r = rumpy.where(cond, x, 0)
        n = np.where(np.arange(5) > 2, np.arange(5), 0)
        assert_eq(r, n)

    def test_where_broadcast(self):
        cond = rumpy.arange(3).reshape([3, 1]) > 0
        x = rumpy.ones([3, 3])
        y = rumpy.zeros([3, 3])
        r = rumpy.where(cond, x, y)
        n = np.where(np.arange(3).reshape(3, 1) > 0, np.ones((3, 3)), np.zeros((3, 3)))
        assert_eq(r, n)

    def test_where_all_true(self):
        cond = rumpy.arange(5) >= 0
        r = rumpy.where(cond, 1, 0)
        n = np.where(np.arange(5) >= 0, 1, 0)
        assert_eq(r, n)

    def test_where_all_false(self):
        cond = rumpy.arange(5) < 0
        r = rumpy.where(cond, 1, 0)
        n = np.where(np.arange(5) < 0, 1, 0)
        assert_eq(r, n)
