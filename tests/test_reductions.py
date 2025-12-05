"""Tests for reduction operations."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


# ============================================================================
# Full reductions (no axis)
# ============================================================================

class TestSum:
    """Test sum reduction."""

    def test_sum_1d(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert r.sum() == pytest.approx(n.sum())

    def test_sum_2d(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert r.sum() == pytest.approx(n.sum())

    def test_sum_view(self):
        r = rp.arange(20)[5:15]
        n = np.arange(20, dtype=np.float64)[5:15]
        assert r.sum() == pytest.approx(n.sum())

    def test_sum_empty(self):
        r = rp.arange(0)
        assert r.sum() == 0.0


class TestProd:
    """Test prod reduction."""

    def test_prod_1d(self):
        r = rp.arange(1, 6)
        n = np.arange(1, 6, dtype=np.float64)
        assert r.prod() == pytest.approx(n.prod())

    def test_prod_2d(self):
        r = rp.arange(1, 7).reshape(2, 3)
        n = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
        assert r.prod() == pytest.approx(n.prod())


class TestMinMax:
    """Test min/max reductions."""

    def test_max_1d(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert r.max() == n.max()

    def test_min_1d(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert r.min() == n.min()

    def test_max_2d(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert r.max() == n.max()

    def test_min_2d(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert r.min() == n.min()

    def test_max_negative(self):
        r = rp.arange(-10, 0)
        n = np.arange(-10, 0, dtype=np.float64)
        assert r.max() == n.max()


class TestMean:
    """Test mean reduction."""

    def test_mean_1d(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert r.mean() == pytest.approx(n.mean())

    def test_mean_2d(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert r.mean() == pytest.approx(n.mean())

    def test_mean_view(self):
        r = rp.arange(20)[5:15]
        n = np.arange(20, dtype=np.float64)[5:15]
        assert r.mean() == pytest.approx(n.mean())

    def test_mean_empty(self):
        r = rp.arange(0)
        assert np.isnan(r.mean())


# ============================================================================
# Axis reductions
# ============================================================================

class TestSumAxis:
    """Test sum along axis."""

    def test_sum_axis0_2d(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.sum(axis=0), n.sum(axis=0))

    def test_sum_axis1_2d(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.sum(axis=1), n.sum(axis=1))

    def test_sum_axis0_3d(self):
        r = rp.arange(24).reshape(2, 3, 4)
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r.sum(axis=0), n.sum(axis=0))

    def test_sum_axis1_3d(self):
        r = rp.arange(24).reshape(2, 3, 4)
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r.sum(axis=1), n.sum(axis=1))

    def test_sum_axis2_3d(self):
        r = rp.arange(24).reshape(2, 3, 4)
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r.sum(axis=2), n.sum(axis=2))

    def test_sum_axis_invalid(self):
        r = rp.arange(12).reshape(3, 4)
        with pytest.raises(ValueError):
            r.sum(axis=2)


class TestMeanAxis:
    """Test mean along axis."""

    def test_mean_axis0_2d(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.mean(axis=0), n.mean(axis=0))

    def test_mean_axis1_2d(self):
        r = rp.arange(12, dtype="float64").reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.mean(axis=1), n.mean(axis=1))


class TestMaxMinAxis:
    """Test max/min along axis."""

    def test_max_axis0(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.max(axis=0), n.max(axis=0))

    def test_max_axis1(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.max(axis=1), n.max(axis=1))

    def test_min_axis0(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.min(axis=0), n.min(axis=0))

    def test_min_axis1(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.min(axis=1), n.min(axis=1))


class TestProdAxis:
    """Test prod along axis."""

    def test_prod_axis0(self):
        r = rp.arange(1, 7).reshape(2, 3)
        n = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
        assert_eq(r.prod(axis=0), n.prod(axis=0))

    def test_prod_axis1(self):
        r = rp.arange(1, 7).reshape(2, 3)
        n = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
        assert_eq(r.prod(axis=1), n.prod(axis=1))


# ============================================================================
# Boolean reductions
# ============================================================================

class TestAll:
    """Test all() boolean reduction."""

    def test_all_true(self):
        r = rp.asarray([1.0, 2.0, 3.0])
        n = np.array([1.0, 2.0, 3.0])
        assert bool(r.all()) == n.all()

    def test_all_false(self):
        r = rp.asarray([1.0, 0.0, 3.0])
        n = np.array([1.0, 0.0, 3.0])
        assert bool(r.all()) == n.all()

    def test_all_empty(self):
        r = rp.asarray([])
        n = np.array([])
        # Empty arrays: numpy returns True for all()
        assert bool(r.all()) == n.all()

    def test_all_2d(self):
        r = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert bool(r.all()) == n.all()

    def test_all_module_func(self):
        r = rp.asarray([1.0, 2.0, 3.0])
        n = np.array([1.0, 2.0, 3.0])
        assert bool(rp.all(r)) == np.all(n)


class TestAny:
    """Test any() boolean reduction."""

    def test_any_true(self):
        r = rp.asarray([0.0, 0.0, 1.0])
        n = np.array([0.0, 0.0, 1.0])
        assert bool(r.any()) == n.any()

    def test_any_false(self):
        r = rp.asarray([0.0, 0.0, 0.0])
        n = np.array([0.0, 0.0, 0.0])
        assert bool(r.any()) == n.any()

    def test_any_empty(self):
        r = rp.asarray([])
        n = np.array([])
        # Empty arrays: numpy returns False for any()
        assert bool(r.any()) == n.any()

    def test_any_2d(self):
        r = rp.asarray([[0.0, 0.0], [0.0, 1.0]])
        n = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert bool(r.any()) == n.any()

    def test_any_module_func(self):
        r = rp.asarray([0.0, 1.0, 0.0])
        n = np.array([0.0, 1.0, 0.0])
        assert bool(rp.any(r)) == np.any(n)


# ============================================================================
# Clip and round
# ============================================================================

class TestClip:
    """Test clip() operation."""

    def test_clip_both_bounds(self):
        r = rp.asarray([1.0, 5.0, 10.0, 15.0, 20.0])
        n = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
        assert_eq(r.clip(5.0, 15.0), n.clip(5.0, 15.0))

    def test_clip_min_only(self):
        r = rp.asarray([1.0, 5.0, 10.0])
        n = np.array([1.0, 5.0, 10.0])
        assert_eq(r.clip(a_min=5.0), n.clip(5.0, None))

    def test_clip_max_only(self):
        r = rp.asarray([1.0, 5.0, 10.0])
        n = np.array([1.0, 5.0, 10.0])
        assert_eq(r.clip(a_max=5.0), n.clip(None, 5.0))

    def test_clip_2d(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.clip(3.0, 8.0), n.clip(3.0, 8.0))

    def test_clip_module_func(self):
        r = rp.asarray([1.0, 5.0, 10.0])
        n = np.array([1.0, 5.0, 10.0])
        assert_eq(rp.clip(r, 3.0, 7.0), np.clip(n, 3.0, 7.0))


class TestRound:
    """Test round() operation."""

    def test_round_default(self):
        # Note: Rust uses "round half away from zero" while NumPy uses
        # "round half to even" (banker's rounding). Test with values that
        # don't hit the 0.5 edge case.
        r = rp.asarray([1.4, 1.6, 2.3, 2.7])
        n = np.array([1.4, 1.6, 2.3, 2.7])
        assert_eq(r.round(), n.round())

    def test_round_decimals(self):
        r = rp.asarray([1.234, 5.678, 9.012])
        n = np.array([1.234, 5.678, 9.012])
        assert_eq(r.round(2), n.round(2))

    def test_round_negative_decimals(self):
        r = rp.asarray([123.0, 456.0, 789.0])
        n = np.array([123.0, 456.0, 789.0])
        assert_eq(r.round(-1), n.round(-1))

    def test_round_2d(self):
        r = rp.asarray([[1.234, 5.678], [9.012, 3.456]])
        n = np.array([[1.234, 5.678], [9.012, 3.456]])
        assert_eq(r.round(1), n.round(1))

    def test_round_module_func(self):
        r = rp.asarray([1.234, 5.678])
        n = np.array([1.234, 5.678])
        assert_eq(rp.round(r, 1), np.round(n, 1))


# ============================================================================
# Cumulative operations
# ============================================================================

class TestCumsum:
    """Test cumsum() operation."""

    def test_cumsum_1d(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        assert_eq(r.cumsum(), n.cumsum())

    def test_cumsum_2d_flat(self):
        r = rp.arange(6).reshape(2, 3)
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(r.cumsum(), n.cumsum())

    def test_cumsum_2d_axis0(self):
        r = rp.arange(6).reshape(2, 3)
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(r.cumsum(axis=0), n.cumsum(axis=0))

    def test_cumsum_2d_axis1(self):
        r = rp.arange(6).reshape(2, 3)
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(r.cumsum(axis=1), n.cumsum(axis=1))

    def test_cumsum_module_func(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        assert_eq(rp.cumsum(r), np.cumsum(n))


class TestCumprod:
    """Test cumprod() operation."""

    def test_cumprod_1d(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert_eq(r.cumprod(), n.cumprod())

    def test_cumprod_2d_flat(self):
        r = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert_eq(r.cumprod(), n.cumprod())

    def test_cumprod_2d_axis0(self):
        r = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert_eq(r.cumprod(axis=0), n.cumprod(axis=0))

    def test_cumprod_2d_axis1(self):
        r = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert_eq(r.cumprod(axis=1), n.cumprod(axis=1))

    def test_cumprod_module_func(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0])
        n = np.array([1.0, 2.0, 3.0, 4.0])
        assert_eq(rp.cumprod(r), np.cumprod(n))


# ============================================================================
# Keepdims tests
# ============================================================================

class TestKeepdims:
    """Test keepdims parameter for reductions."""

    def test_sum_keepdims_axis(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.sum(axis=0, keepdims=True), n.sum(axis=0, keepdims=True))
        assert_eq(r.sum(axis=1, keepdims=True), n.sum(axis=1, keepdims=True))

    def test_sum_keepdims_full(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        result = r.sum(keepdims=True)
        expected = n.sum(keepdims=True)
        assert result.shape == expected.shape
        # Check value
        assert float(result[0, 0]) == pytest.approx(float(expected[0, 0]))

    def test_mean_keepdims_axis(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.mean(axis=0, keepdims=True), n.mean(axis=0, keepdims=True))
        assert_eq(r.mean(axis=1, keepdims=True), n.mean(axis=1, keepdims=True))

    def test_max_keepdims_axis(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.max(axis=0, keepdims=True), n.max(axis=0, keepdims=True))
        assert_eq(r.max(axis=1, keepdims=True), n.max(axis=1, keepdims=True))

    def test_min_keepdims_axis(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.min(axis=0, keepdims=True), n.min(axis=0, keepdims=True))
        assert_eq(r.min(axis=1, keepdims=True), n.min(axis=1, keepdims=True))

    def test_prod_keepdims_axis(self):
        r = rp.asarray([[1.0, 2.0], [3.0, 4.0]])
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_eq(r.prod(axis=0, keepdims=True), n.prod(axis=0, keepdims=True))
        assert_eq(r.prod(axis=1, keepdims=True), n.prod(axis=1, keepdims=True))

    def test_var_keepdims_axis(self):
        r = rp.arange(12, dtype="float64").reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.var(axis=0, keepdims=True), n.var(axis=0, keepdims=True))
        assert_eq(r.var(axis=1, keepdims=True), n.var(axis=1, keepdims=True))

    def test_std_keepdims_axis(self):
        r = rp.arange(12, dtype="float64").reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r.std(axis=0, keepdims=True), n.std(axis=0, keepdims=True))
        assert_eq(r.std(axis=1, keepdims=True), n.std(axis=1, keepdims=True))

    def test_keepdims_3d(self):
        """Test keepdims on 3D array."""
        r = rp.arange(24).reshape(2, 3, 4)
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r.sum(axis=1, keepdims=True), n.sum(axis=1, keepdims=True))
        assert r.sum(axis=1, keepdims=True).shape == (2, 1, 4)
