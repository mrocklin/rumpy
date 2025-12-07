"""Tests for NaN-aware reduction operations."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


# ============================================================================
# nansum
# ============================================================================

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


# ============================================================================
# nanprod
# ============================================================================

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


# ============================================================================
# nanmean
# ============================================================================

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


# ============================================================================
# nanstd
# ============================================================================

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


# ============================================================================
# nanvar
# ============================================================================

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


# ============================================================================
# nanmin
# ============================================================================

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


# ============================================================================
# nanmax
# ============================================================================

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


# ============================================================================
# nanargmin
# ============================================================================

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


# ============================================================================
# nanargmax
# ============================================================================

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


# ============================================================================
# Edge cases
# ============================================================================

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
