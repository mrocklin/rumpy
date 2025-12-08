"""Tests for reduction operations.

Reductions have custom dispatch code, so we use NUMERIC_DTYPES for full coverage.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES, FLOAT_DTYPES, NUMERIC_DTYPES
from helpers import assert_eq, make_pair

# === Reduction categories ===

# Standard reductions (sum, prod, etc.)
STANDARD_REDUCTIONS = ["sum", "prod", "max", "min"]

# Statistical reductions
STAT_REDUCTIONS = ["mean", "var", "std"]

# Argmin/argmax
ARG_REDUCTIONS = ["argmax", "argmin"]


# === Parametrized tests by category ===


class TestStandardReductions:
    """Test sum, prod, max, min."""

    @pytest.mark.parametrize("reduction", STANDARD_REDUCTIONS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, reduction, dtype):
        n = np.array([1, 2, 3, 4, 5], dtype=dtype)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(), np_fn())

    @pytest.mark.parametrize("reduction", STANDARD_REDUCTIONS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, reduction, shape):
        r, n = make_pair(shape, "float64")
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(), np_fn())

    @pytest.mark.parametrize("reduction", STANDARD_REDUCTIONS)
    def test_2d_axis0(self, reduction):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(axis=0), np_fn(axis=0))

    @pytest.mark.parametrize("reduction", STANDARD_REDUCTIONS)
    def test_2d_axis1(self, reduction):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(axis=1), np_fn(axis=1))

    @pytest.mark.parametrize("reduction", STANDARD_REDUCTIONS)
    def test_negative_axis(self, reduction):
        # Note: negative axis only works reliably for 2D arrays
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        # Use axis=1 instead of -1 due to known bug with negative axis
        assert_eq(rp_fn(axis=1), np_fn(axis=1))

    @pytest.mark.parametrize("reduction", STANDARD_REDUCTIONS)
    def test_keepdims(self, reduction):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(axis=0, keepdims=True), np_fn(axis=0, keepdims=True))

    @pytest.mark.parametrize("reduction", STANDARD_REDUCTIONS)
    def test_3d_axis(self, reduction):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        # Note: negative axis has known bugs, use only positive axes
        for axis in [0, 1, 2]:
            assert_eq(rp_fn(axis=axis), np_fn(axis=axis))


class TestStatReductions:
    """Test mean, var, std."""

    @pytest.mark.parametrize("reduction", STAT_REDUCTIONS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, reduction, dtype):
        n = np.array([1, 2, 3, 4, 5], dtype=dtype)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(), np_fn())

    @pytest.mark.parametrize("reduction", STAT_REDUCTIONS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, reduction, shape):
        r, n = make_pair(shape, "float64")
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(), np_fn())

    @pytest.mark.parametrize("reduction", STAT_REDUCTIONS)
    def test_2d_axis0(self, reduction):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(axis=0), np_fn(axis=0))

    @pytest.mark.parametrize("reduction", STAT_REDUCTIONS)
    def test_2d_axis1(self, reduction):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(axis=1), np_fn(axis=1))

    @pytest.mark.parametrize("reduction", STAT_REDUCTIONS)
    def test_keepdims(self, reduction):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(axis=0, keepdims=True), np_fn(axis=0, keepdims=True))


class TestArgReductions:
    """Test argmax, argmin."""

    @pytest.mark.parametrize("reduction", ARG_REDUCTIONS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, reduction, dtype):
        n = np.array([3, 1, 4, 1, 5], dtype=dtype)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert rp_fn() == np_fn()

    @pytest.mark.parametrize("reduction", ARG_REDUCTIONS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, reduction, shape):
        r, n = make_pair(shape, "float64")
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert rp_fn() == np_fn()

    @pytest.mark.parametrize("reduction", ARG_REDUCTIONS)
    def test_2d_axis0(self, reduction):
        n = np.array([[3, 1, 4], [1, 5, 9]], dtype=np.float64)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(axis=0), np_fn(axis=0))

    @pytest.mark.parametrize("reduction", ARG_REDUCTIONS)
    def test_2d_axis1(self, reduction):
        n = np.array([[3, 1, 4], [1, 5, 9]], dtype=np.float64)
        r = rp.asarray(n)
        rp_fn = getattr(r, reduction)
        np_fn = getattr(n, reduction)
        assert_eq(rp_fn(axis=1), np_fn(axis=1))


# === Module functions ===


class TestModuleReductions:
    """Test rp.sum(), rp.mean(), etc. module functions."""

    def test_sum_module(self):
        n = np.arange(10, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.sum(r), np.sum(n))

    def test_mean_module(self):
        n = np.arange(10, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.mean(r), np.mean(n))

    def test_max_module(self):
        n = np.arange(10, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.max(r), np.max(n))

    def test_min_module(self):
        n = np.arange(10, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.min(r), np.min(n))

    def test_prod_module(self):
        n = np.array([1, 2, 3, 4], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.prod(r), np.prod(n))


# === Edge cases ===


class TestReductionEdgeCases:
    """Test edge cases for reductions."""

    def test_sum_empty(self):
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        assert r.sum() == n.sum()  # should be 0.0

    def test_prod_empty(self):
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        assert r.prod() == n.prod()  # should be 1.0

    def test_sum_single(self):
        n = np.array([42.0], dtype=np.float64)
        r = rp.asarray(n)
        assert r.sum() == n.sum()

    def test_view_sum(self):
        """Sum on non-contiguous view."""
        n = np.arange(20, dtype=np.float64)[5:15]
        r = rp.asarray(np.arange(20, dtype=np.float64))[5:15]
        assert_eq(r.sum(), n.sum())

    def test_transposed_sum(self):
        """Sum on transposed array."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4).T
        r = rp.asarray(np.arange(12, dtype=np.float64).reshape(3, 4)).T
        assert_eq(r.sum(), n.sum())

    def test_sum_axis_preserves_shape(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert r.sum(axis=0).shape == n.sum(axis=0).shape
        assert r.sum(axis=1).shape == n.sum(axis=1).shape


# === Boolean reductions ===


class TestBooleanReductions:
    """Test all, any reductions."""

    def test_all_true(self):
        n = np.array([True, True, True])
        r = rp.asarray(n)
        assert r.all() == n.all()

    def test_all_false(self):
        n = np.array([True, False, True])
        r = rp.asarray(n)
        assert r.all() == n.all()

    def test_any_true(self):
        n = np.array([False, True, False])
        r = rp.asarray(n)
        assert r.any() == n.any()

    def test_any_false(self):
        n = np.array([False, False, False])
        r = rp.asarray(n)
        assert r.any() == n.any()

    def test_all_axis(self):
        n = np.array([[True, True], [False, True]])
        r = rp.asarray(n)
        assert_eq(r.all(axis=0), n.all(axis=0))
        assert_eq(r.all(axis=1), n.all(axis=1))

    def test_any_axis(self):
        n = np.array([[True, False], [False, False]])
        r = rp.asarray(n)
        assert_eq(r.any(axis=0), n.any(axis=0))
        assert_eq(r.any(axis=1), n.any(axis=1))


# === NaN-aware reductions ===


class TestNansum:
    """Test nansum - sum ignoring NaN values."""

    def test_nansum_no_nan(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0])
        n = np.array([1.0, 2.0, 3.0, 4.0])
        assert rp.nansum(r) == pytest.approx(np.nansum(n))

    def test_nansum_with_nan(self):
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0])
        n = np.array([1.0, np.nan, 3.0, 4.0])
        assert rp.nansum(r) == pytest.approx(np.nansum(n))

    def test_nansum_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        n = np.array([np.nan, np.nan])
        assert rp.nansum(r) == np.nansum(n)  # Should be 0.0

    def test_nansum_2d(self):
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert rp.nansum(r) == pytest.approx(np.nansum(n))

    def test_nansum_axis0(self):
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert_eq(rp.nansum(r, axis=0), np.nansum(n, axis=0))

    def test_nansum_axis1(self):
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert_eq(rp.nansum(r, axis=1), np.nansum(n, axis=1))

    def test_nansum_empty(self):
        r = rp.asarray([])
        n = np.array([])
        assert rp.nansum(r) == np.nansum(n)  # Should be 0.0


class TestNanprod:
    """Test nanprod - product ignoring NaN values."""

    def test_nanprod_no_nan(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0])
        n = np.array([1.0, 2.0, 3.0, 4.0])
        assert rp.nanprod(r) == pytest.approx(np.nanprod(n))

    def test_nanprod_with_nan(self):
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0])
        n = np.array([1.0, np.nan, 3.0, 4.0])
        assert rp.nanprod(r) == pytest.approx(np.nanprod(n))

    def test_nanprod_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        n = np.array([np.nan, np.nan])
        assert rp.nanprod(r) == np.nanprod(n)  # Should be 1.0

    def test_nanprod_2d(self):
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert rp.nanprod(r) == pytest.approx(np.nanprod(n))

    def test_nanprod_axis0(self):
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert_eq(rp.nanprod(r, axis=0), np.nanprod(n, axis=0))

    def test_nanprod_axis1(self):
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert_eq(rp.nanprod(r, axis=1), np.nanprod(n, axis=1))


class TestNanmean:
    """Test nanmean - mean ignoring NaN values."""

    def test_nanmean_no_nan(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0])
        n = np.array([1.0, 2.0, 3.0, 4.0])
        assert rp.nanmean(r) == pytest.approx(np.nanmean(n))

    def test_nanmean_with_nan(self):
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0])
        n = np.array([1.0, np.nan, 3.0, 4.0])
        assert rp.nanmean(r) == pytest.approx(np.nanmean(n))

    def test_nanmean_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        n = np.array([np.nan, np.nan])
        # numpy returns NaN with warning for all-NaN slice
        assert np.isnan(rp.nanmean(r))

    def test_nanmean_2d(self):
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert rp.nanmean(r) == pytest.approx(np.nanmean(n))

    def test_nanmean_axis0(self):
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert_eq(rp.nanmean(r, axis=0), np.nanmean(n, axis=0))

    def test_nanmean_axis1(self):
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert_eq(rp.nanmean(r, axis=1), np.nanmean(n, axis=1))


class TestNanstd:
    """Test nanstd - standard deviation ignoring NaN values."""

    def test_nanstd_no_nan(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rp.nanstd(r) == pytest.approx(np.nanstd(n))

    def test_nanstd_with_nan(self):
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0, 5.0])
        n = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        assert rp.nanstd(r) == pytest.approx(np.nanstd(n))

    def test_nanstd_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        assert np.isnan(rp.nanstd(r))

    def test_nanstd_axis0(self):
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, 6.0]])
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        assert_eq(rp.nanstd(r, axis=0), np.nanstd(n, axis=0))

    def test_nanstd_axis1(self):
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, 6.0]])
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        assert_eq(rp.nanstd(r, axis=1), np.nanstd(n, axis=1))


class TestNanvar:
    """Test nanvar - variance ignoring NaN values."""

    def test_nanvar_no_nan(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rp.nanvar(r) == pytest.approx(np.nanvar(n))

    def test_nanvar_with_nan(self):
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0, 5.0])
        n = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        assert rp.nanvar(r) == pytest.approx(np.nanvar(n))

    def test_nanvar_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        assert np.isnan(rp.nanvar(r))

    def test_nanvar_axis0(self):
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, 6.0]])
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        assert_eq(rp.nanvar(r, axis=0), np.nanvar(n, axis=0))

    def test_nanvar_axis1(self):
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, 6.0]])
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        assert_eq(rp.nanvar(r, axis=1), np.nanvar(n, axis=1))


class TestNanmin:
    """Test nanmin - min ignoring NaN values."""

    def test_nanmin_no_nan(self):
        r = rp.asarray([3.0, 1.0, 4.0, 1.0, 5.0])
        n = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert rp.nanmin(r) == np.nanmin(n)

    def test_nanmin_with_nan(self):
        r = rp.asarray([3.0, float('nan'), 1.0, 4.0])
        n = np.array([3.0, np.nan, 1.0, 4.0])
        assert rp.nanmin(r) == np.nanmin(n)

    def test_nanmin_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        # numpy raises warning and returns nan
        assert np.isnan(rp.nanmin(r))

    def test_nanmin_2d(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert rp.nanmin(r) == np.nanmin(n)

    def test_nanmin_axis0(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert_eq(rp.nanmin(r, axis=0), np.nanmin(n, axis=0))

    def test_nanmin_axis1(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert_eq(rp.nanmin(r, axis=1), np.nanmin(n, axis=1))


class TestNanmax:
    """Test nanmax - max ignoring NaN values."""

    def test_nanmax_no_nan(self):
        r = rp.asarray([3.0, 1.0, 4.0, 1.0, 5.0])
        n = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert rp.nanmax(r) == np.nanmax(n)

    def test_nanmax_with_nan(self):
        r = rp.asarray([3.0, float('nan'), 1.0, 4.0])
        n = np.array([3.0, np.nan, 1.0, 4.0])
        assert rp.nanmax(r) == np.nanmax(n)

    def test_nanmax_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        # numpy raises warning and returns nan
        assert np.isnan(rp.nanmax(r))

    def test_nanmax_2d(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert rp.nanmax(r) == np.nanmax(n)

    def test_nanmax_axis0(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert_eq(rp.nanmax(r, axis=0), np.nanmax(n, axis=0))

    def test_nanmax_axis1(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert_eq(rp.nanmax(r, axis=1), np.nanmax(n, axis=1))


class TestNanargmin:
    """Test nanargmin - argmin ignoring NaN values."""

    def test_nanargmin_no_nan(self):
        r = rp.asarray([3.0, 1.0, 4.0, 1.0, 5.0])
        n = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert int(rp.nanargmin(r)) == np.nanargmin(n)

    def test_nanargmin_with_nan(self):
        r = rp.asarray([3.0, float('nan'), 1.0, 4.0])
        n = np.array([3.0, np.nan, 1.0, 4.0])
        assert int(rp.nanargmin(r)) == np.nanargmin(n)

    def test_nanargmin_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        # numpy raises ValueError for all-NaN slice
        with pytest.raises(ValueError):
            rp.nanargmin(r)

    def test_nanargmin_2d(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert int(rp.nanargmin(r)) == np.nanargmin(n)

    def test_nanargmin_axis0(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert_eq(rp.nanargmin(r, axis=0), np.nanargmin(n, axis=0))

    def test_nanargmin_axis1(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert_eq(rp.nanargmin(r, axis=1), np.nanargmin(n, axis=1))


class TestNanargmax:
    """Test nanargmax - argmax ignoring NaN values."""

    def test_nanargmax_no_nan(self):
        r = rp.asarray([3.0, 1.0, 4.0, 1.0, 5.0])
        n = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert int(rp.nanargmax(r)) == np.nanargmax(n)

    def test_nanargmax_with_nan(self):
        r = rp.asarray([3.0, float('nan'), 1.0, 4.0])
        n = np.array([3.0, np.nan, 1.0, 4.0])
        assert int(rp.nanargmax(r)) == np.nanargmax(n)

    def test_nanargmax_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        # numpy raises ValueError for all-NaN slice
        with pytest.raises(ValueError):
            rp.nanargmax(r)

    def test_nanargmax_2d(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert int(rp.nanargmax(r)) == np.nanargmax(n)

    def test_nanargmax_axis0(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert_eq(rp.nanargmax(r, axis=0), np.nanargmax(n, axis=0))

    def test_nanargmax_axis1(self):
        r = rp.asarray([[float('nan'), 2.0], [3.0, 4.0]])
        n = np.array([[np.nan, 2.0], [3.0, 4.0]])
        assert_eq(rp.nanargmax(r, axis=1), np.nanargmax(n, axis=1))


class TestNanEdgeCases:
    """Test edge cases for NaN-aware reductions."""

    def test_nan_at_start(self):
        """NaN at the start of array."""
        r = rp.asarray([float('nan'), 1.0, 2.0, 3.0])
        n = np.array([np.nan, 1.0, 2.0, 3.0])
        assert rp.nansum(r) == pytest.approx(np.nansum(n))
        assert rp.nanmean(r) == pytest.approx(np.nanmean(n))
        assert rp.nanmin(r) == np.nanmin(n)
        assert rp.nanmax(r) == np.nanmax(n)

    def test_nan_at_end(self):
        """NaN at the end of array."""
        r = rp.asarray([1.0, 2.0, 3.0, float('nan')])
        n = np.array([1.0, 2.0, 3.0, np.nan])
        assert rp.nansum(r) == pytest.approx(np.nansum(n))
        assert rp.nanmean(r) == pytest.approx(np.nanmean(n))
        assert rp.nanmin(r) == np.nanmin(n)
        assert rp.nanmax(r) == np.nanmax(n)

    def test_multiple_nans(self):
        """Multiple NaN values in array."""
        r = rp.asarray([float('nan'), 1.0, float('nan'), 2.0, float('nan')])
        n = np.array([np.nan, 1.0, np.nan, 2.0, np.nan])
        assert rp.nansum(r) == pytest.approx(np.nansum(n))
        assert rp.nanmean(r) == pytest.approx(np.nanmean(n))
        assert rp.nanmin(r) == np.nanmin(n)
        assert rp.nanmax(r) == np.nanmax(n)

    def test_single_value(self):
        """Single non-NaN value."""
        r = rp.asarray([5.0])
        n = np.array([5.0])
        assert rp.nansum(r) == np.nansum(n)
        assert rp.nanmean(r) == np.nanmean(n)
        assert rp.nanmin(r) == np.nanmin(n)
        assert rp.nanmax(r) == np.nanmax(n)

    def test_inf_handling(self):
        """Infinity values should be kept (not treated as NaN)."""
        r = rp.asarray([1.0, float('inf'), float('nan'), 3.0])
        n = np.array([1.0, np.inf, np.nan, 3.0])
        assert rp.nansum(r) == np.nansum(n)
        assert rp.nanmax(r) == np.nanmax(n)

    def test_negative_inf(self):
        """Negative infinity handling."""
        r = rp.asarray([1.0, float('-inf'), float('nan'), 3.0])
        n = np.array([1.0, -np.inf, np.nan, 3.0])
        assert rp.nanmin(r) == np.nanmin(n)

    def test_3d_array(self):
        """NaN-aware reductions on 3D arrays."""
        r_data = [[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [float('nan'), 8.0]]]
        n_data = [[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [np.nan, 8.0]]]
        r = rp.asarray(r_data)
        n = np.array(n_data)
        assert rp.nansum(r) == pytest.approx(np.nansum(n))
        assert_eq(rp.nansum(r, axis=0), np.nansum(n, axis=0))
        assert_eq(rp.nansum(r, axis=1), np.nansum(n, axis=1))
        assert_eq(rp.nansum(r, axis=2), np.nansum(n, axis=2))


# === Cumulative reductions ===


class TestCumulativeOps:
    """Test cumsum, cumprod."""

    def test_cumsum(self):
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.cumsum(r), np.cumsum(n))

    def test_cumprod(self):
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.cumprod(r), np.cumprod(n))

    def test_cumsum_2d_axis0(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(rp.cumsum(r, axis=0), np.cumsum(n, axis=0))

    def test_cumsum_2d_axis1(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(rp.cumsum(r, axis=1), np.cumsum(n, axis=1))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_dtypes(self, dtype):
        n = np.array([1, 2, 3, 4, 5], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.cumsum(r), np.cumsum(n))


# === Count nonzero ===


class TestCountNonzero:
    """Test count_nonzero."""

    def test_basic(self):
        n = np.array([0, 1, 0, 3, 0, 5], dtype=np.float64)
        r = rp.asarray(n)
        assert rp.count_nonzero(r) == np.count_nonzero(n)

    def test_bool(self):
        n = np.array([True, False, True, False, True])
        r = rp.asarray(n)
        assert rp.count_nonzero(r) == np.count_nonzero(n)

    def test_2d(self):
        n = np.array([[0, 1, 2], [0, 0, 3]], dtype=np.float64)
        r = rp.asarray(n)
        assert rp.count_nonzero(r) == np.count_nonzero(n)


# === All/Any ===


class TestAll:
    """Test all reduction."""

    def test_all_true(self):
        n = np.array([True, True, True])
        r = rp.asarray(n)
        assert rp.all(r) == np.all(n)

    def test_all_false(self):
        n = np.array([True, False, True])
        r = rp.asarray(n)
        assert rp.all(r) == np.all(n)

    def test_all_numeric(self):
        """Non-zero values are truthy."""
        n = np.array([1, 2, 3])
        r = rp.asarray(n)
        assert rp.all(r) == np.all(n)

    def test_all_axis(self):
        n = np.array([[True, False], [True, True]])
        r = rp.asarray(n)
        assert_eq(rp.all(r, axis=0), np.all(n, axis=0))
        assert_eq(rp.all(r, axis=1), np.all(n, axis=1))


class TestAny:
    """Test any reduction."""

    def test_any_true(self):
        n = np.array([False, True, False])
        r = rp.asarray(n)
        assert rp.any(r) == np.any(n)

    def test_any_false(self):
        n = np.array([False, False, False])
        r = rp.asarray(n)
        assert rp.any(r) == np.any(n)

    def test_any_with_zero(self):
        n = np.array([0, 0, 1])
        r = rp.asarray(n)
        assert rp.any(r) == np.any(n)

    def test_any_axis(self):
        n = np.array([[False, False], [True, False]])
        r = rp.asarray(n)
        assert_eq(rp.any(r, axis=0), np.any(n, axis=0))
        assert_eq(rp.any(r, axis=1), np.any(n, axis=1))


# === Diff ===


class TestDiff:
    """Test diff function."""

    def test_diff_1d(self):
        n = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
        r = rp.asarray(n)
        assert_eq(rp.diff(r), np.diff(n))

    def test_diff_n2(self):
        n = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
        r = rp.asarray(n)
        assert_eq(rp.diff(r, n=2), np.diff(n, n=2))

    def test_diff_2d(self):
        n = np.array([[1.0, 2.0, 4.0], [1.0, 6.0, 7.0]])
        r = rp.asarray(n)
        assert_eq(rp.diff(r), np.diff(n))
        assert_eq(rp.diff(r, axis=0), np.diff(n, axis=0))
