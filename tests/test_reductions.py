"""Tests for reduction operations."""

import numpy as np
import pytest

import rumpy
from helpers import assert_eq


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
