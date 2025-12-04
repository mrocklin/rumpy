"""Tests for reduction operations."""

import numpy as np
import pytest

import rumpy
from helpers import assert_eq


# ============================================================================
# Full reductions (no axis)
# ============================================================================

class TestSum:
    """Test sum reduction."""

    def test_sum_1d(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert r.sum() == pytest.approx(n.sum())

    def test_sum_2d(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert r.sum() == pytest.approx(n.sum())

    def test_sum_view(self):
        r = rumpy.arange(20)[5:15]
        n = np.arange(20, dtype=np.float64)[5:15]
        assert r.sum() == pytest.approx(n.sum())

    def test_sum_empty(self):
        r = rumpy.arange(0)
        assert r.sum() == 0.0


class TestProd:
    """Test prod reduction."""

    def test_prod_1d(self):
        r = rumpy.arange(1, 6)
        n = np.arange(1, 6, dtype=np.float64)
        assert r.prod() == pytest.approx(n.prod())

    def test_prod_2d(self):
        r = rumpy.arange(1, 7).reshape([2, 3])
        n = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
        assert r.prod() == pytest.approx(n.prod())


class TestMinMax:
    """Test min/max reductions."""

    def test_max_1d(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert r.max() == n.max()

    def test_min_1d(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert r.min() == n.min()

    def test_max_2d(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert r.max() == n.max()

    def test_min_2d(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert r.min() == n.min()

    def test_max_negative(self):
        r = rumpy.arange(-10, 0)
        n = np.arange(-10, 0, dtype=np.float64)
        assert r.max() == n.max()


class TestMean:
    """Test mean reduction."""

    def test_mean_1d(self):
        r = rumpy.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert r.mean() == pytest.approx(n.mean())

    def test_mean_2d(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert r.mean() == pytest.approx(n.mean())

    def test_mean_view(self):
        r = rumpy.arange(20)[5:15]
        n = np.arange(20, dtype=np.float64)[5:15]
        assert r.mean() == pytest.approx(n.mean())

    def test_mean_empty(self):
        r = rumpy.arange(0)
        assert np.isnan(r.mean())


# ============================================================================
# Axis reductions
# ============================================================================

class TestSumAxis:
    """Test sum along axis."""

    def test_sum_axis0_2d(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.sum(axis=0), n.sum(axis=0))

    def test_sum_axis1_2d(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.sum(axis=1), n.sum(axis=1))

    def test_sum_axis0_3d(self):
        r = rumpy.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r.sum(axis=0), n.sum(axis=0))

    def test_sum_axis1_3d(self):
        r = rumpy.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r.sum(axis=1), n.sum(axis=1))

    def test_sum_axis2_3d(self):
        r = rumpy.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r.sum(axis=2), n.sum(axis=2))

    def test_sum_axis_invalid(self):
        r = rumpy.arange(12).reshape([3, 4])
        with pytest.raises(ValueError):
            r.sum(axis=2)


class TestMeanAxis:
    """Test mean along axis."""

    def test_mean_axis0_2d(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.mean(axis=0), n.mean(axis=0))

    def test_mean_axis1_2d(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.mean(axis=1), n.mean(axis=1))


class TestMaxMinAxis:
    """Test max/min along axis."""

    def test_max_axis0(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.max(axis=0), n.max(axis=0))

    def test_max_axis1(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.max(axis=1), n.max(axis=1))

    def test_min_axis0(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.min(axis=0), n.min(axis=0))

    def test_min_axis1(self):
        r = rumpy.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.min(axis=1), n.min(axis=1))


class TestProdAxis:
    """Test prod along axis."""

    def test_prod_axis0(self):
        r = rumpy.arange(1, 7).reshape([2, 3])
        n = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
        assert_eq(r.prod(axis=0), n.prod(axis=0))

    def test_prod_axis1(self):
        r = rumpy.arange(1, 7).reshape([2, 3])
        n = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
        assert_eq(r.prod(axis=1), n.prod(axis=1))
