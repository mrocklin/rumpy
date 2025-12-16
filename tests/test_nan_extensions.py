"""Tests for NaN-aware extension functions (Stream 31).

Tests nanmedian, nanpercentile, nanquantile, nancumsum, nancumprod.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


# === nanmedian ===


class TestNanmedian:
    """Test nanmedian - median ignoring NaN values."""

    def test_nanmedian_no_nan(self):
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = rp.asarray(n)
        assert rp.nanmedian(r) == pytest.approx(np.nanmedian(n))

    def test_nanmedian_with_nan(self):
        n = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0, 5.0])
        assert rp.nanmedian(r) == pytest.approx(np.nanmedian(n))

    def test_nanmedian_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        # numpy returns nan (with warning) for all-NaN slice
        assert np.isnan(rp.nanmedian(r))

    def test_nanmedian_even_count(self):
        """Even number of non-NaN values - average of middle two."""
        n = np.array([1.0, np.nan, 3.0, 5.0, np.nan, 7.0])
        r = rp.asarray([1.0, float('nan'), 3.0, 5.0, float('nan'), 7.0])
        assert rp.nanmedian(r) == pytest.approx(np.nanmedian(n))

    def test_nanmedian_2d(self):
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        assert rp.nanmedian(r) == pytest.approx(np.nanmedian(n))

    def test_nanmedian_axis0(self):
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        assert_eq(rp.nanmedian(r, axis=0), np.nanmedian(n, axis=0))

    def test_nanmedian_axis1(self):
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        assert_eq(rp.nanmedian(r, axis=1), np.nanmedian(n, axis=1))

    def test_nanmedian_single_value(self):
        r = rp.asarray([5.0])
        n = np.array([5.0])
        assert rp.nanmedian(r) == np.nanmedian(n)

    def test_nanmedian_single_nan(self):
        r = rp.asarray([float('nan')])
        assert np.isnan(rp.nanmedian(r))


# === nanpercentile ===


class TestNanpercentile:
    """Test nanpercentile - percentile ignoring NaN values."""

    def test_nanpercentile_no_nan(self):
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = rp.asarray(n)
        # rumpy returns array, numpy returns scalar for single q
        assert abs(float(rp.nanpercentile(r, 50)[0]) - np.nanpercentile(n, 50)) < 1e-10

    def test_nanpercentile_with_nan(self):
        n = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0, 5.0])
        assert abs(float(rp.nanpercentile(r, 50)[0]) - np.nanpercentile(n, 50)) < 1e-10

    def test_nanpercentile_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        result = rp.nanpercentile(r, 50)
        assert np.isnan(float(result[0]))

    def test_nanpercentile_multiple_q(self):
        n = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0, 5.0])
        assert_eq(rp.nanpercentile(r, [25, 50, 75]), np.nanpercentile(n, [25, 50, 75]))

    def test_nanpercentile_extremes(self):
        n = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0, 5.0])
        assert abs(float(rp.nanpercentile(r, 0)[0]) - np.nanpercentile(n, 0)) < 1e-10
        assert abs(float(rp.nanpercentile(r, 100)[0]) - np.nanpercentile(n, 100)) < 1e-10

    def test_nanpercentile_2d(self):
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        assert abs(float(rp.nanpercentile(r, 50)[0]) - np.nanpercentile(n, 50)) < 1e-10

    def test_nanpercentile_axis0(self):
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        assert_eq(rp.nanpercentile(r, 50, axis=0), np.nanpercentile(n, 50, axis=0))

    def test_nanpercentile_axis1(self):
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        assert_eq(rp.nanpercentile(r, 50, axis=1), np.nanpercentile(n, 50, axis=1))


# === nanquantile ===


class TestNanquantile:
    """Test nanquantile - quantile ignoring NaN values."""

    def test_nanquantile_no_nan(self):
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = rp.asarray(n)
        # rumpy returns array, numpy returns scalar for single q
        assert abs(float(rp.nanquantile(r, 0.5)[0]) - np.nanquantile(n, 0.5)) < 1e-10

    def test_nanquantile_with_nan(self):
        n = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0, 5.0])
        assert abs(float(rp.nanquantile(r, 0.5)[0]) - np.nanquantile(n, 0.5)) < 1e-10

    def test_nanquantile_all_nan(self):
        r = rp.asarray([float('nan'), float('nan')])
        result = rp.nanquantile(r, 0.5)
        assert np.isnan(float(result[0]))

    def test_nanquantile_multiple_q(self):
        n = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0, 5.0])
        assert_eq(rp.nanquantile(r, [0.25, 0.5, 0.75]), np.nanquantile(n, [0.25, 0.5, 0.75]))

    def test_nanquantile_extremes(self):
        n = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0, 5.0])
        assert abs(float(rp.nanquantile(r, 0.0)[0]) - np.nanquantile(n, 0.0)) < 1e-10
        assert abs(float(rp.nanquantile(r, 1.0)[0]) - np.nanquantile(n, 1.0)) < 1e-10

    def test_nanquantile_2d(self):
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        assert abs(float(rp.nanquantile(r, 0.5)[0]) - np.nanquantile(n, 0.5)) < 1e-10

    def test_nanquantile_axis0(self):
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        assert_eq(rp.nanquantile(r, 0.5, axis=0), np.nanquantile(n, 0.5, axis=0))

    def test_nanquantile_axis1(self):
        n = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        r = rp.asarray([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        assert_eq(rp.nanquantile(r, 0.5, axis=1), np.nanquantile(n, 0.5, axis=1))


# === nancumsum ===


class TestNancumsum:
    """Test nancumsum - cumulative sum treating NaN as zero."""

    def test_nancumsum_no_nan(self):
        n = np.array([1.0, 2.0, 3.0, 4.0])
        r = rp.asarray(n)
        assert_eq(rp.nancumsum(r), np.nancumsum(n))

    def test_nancumsum_with_nan(self):
        n = np.array([1.0, np.nan, 3.0, 4.0])
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0])
        assert_eq(rp.nancumsum(r), np.nancumsum(n))

    def test_nancumsum_all_nan(self):
        n = np.array([np.nan, np.nan, np.nan])
        r = rp.asarray([float('nan'), float('nan'), float('nan')])
        assert_eq(rp.nancumsum(r), np.nancumsum(n))

    def test_nancumsum_2d(self):
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        assert_eq(rp.nancumsum(r), np.nancumsum(n))

    def test_nancumsum_axis0(self):
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        assert_eq(rp.nancumsum(r, axis=0), np.nancumsum(n, axis=0))

    def test_nancumsum_axis1(self):
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        assert_eq(rp.nancumsum(r, axis=1), np.nancumsum(n, axis=1))

    def test_nancumsum_nan_at_start(self):
        n = np.array([np.nan, 1.0, 2.0, 3.0])
        r = rp.asarray([float('nan'), 1.0, 2.0, 3.0])
        assert_eq(rp.nancumsum(r), np.nancumsum(n))

    def test_nancumsum_nan_at_end(self):
        n = np.array([1.0, 2.0, 3.0, np.nan])
        r = rp.asarray([1.0, 2.0, 3.0, float('nan')])
        assert_eq(rp.nancumsum(r), np.nancumsum(n))

    def test_nancumsum_empty(self):
        n = np.array([])
        r = rp.asarray([])
        assert_eq(rp.nancumsum(r), np.nancumsum(n))


# === nancumprod ===


class TestNancumprod:
    """Test nancumprod - cumulative product treating NaN as one."""

    def test_nancumprod_no_nan(self):
        n = np.array([1.0, 2.0, 3.0, 4.0])
        r = rp.asarray(n)
        assert_eq(rp.nancumprod(r), np.nancumprod(n))

    def test_nancumprod_with_nan(self):
        n = np.array([1.0, np.nan, 3.0, 4.0])
        r = rp.asarray([1.0, float('nan'), 3.0, 4.0])
        assert_eq(rp.nancumprod(r), np.nancumprod(n))

    def test_nancumprod_all_nan(self):
        n = np.array([np.nan, np.nan, np.nan])
        r = rp.asarray([float('nan'), float('nan'), float('nan')])
        assert_eq(rp.nancumprod(r), np.nancumprod(n))

    def test_nancumprod_2d(self):
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        assert_eq(rp.nancumprod(r), np.nancumprod(n))

    def test_nancumprod_axis0(self):
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        assert_eq(rp.nancumprod(r, axis=0), np.nancumprod(n, axis=0))

    def test_nancumprod_axis1(self):
        n = np.array([[1.0, np.nan], [3.0, 4.0]])
        r = rp.asarray([[1.0, float('nan')], [3.0, 4.0]])
        assert_eq(rp.nancumprod(r, axis=1), np.nancumprod(n, axis=1))

    def test_nancumprod_nan_at_start(self):
        n = np.array([np.nan, 2.0, 3.0, 4.0])
        r = rp.asarray([float('nan'), 2.0, 3.0, 4.0])
        assert_eq(rp.nancumprod(r), np.nancumprod(n))

    def test_nancumprod_nan_at_end(self):
        n = np.array([1.0, 2.0, 3.0, np.nan])
        r = rp.asarray([1.0, 2.0, 3.0, float('nan')])
        assert_eq(rp.nancumprod(r), np.nancumprod(n))

    def test_nancumprod_empty(self):
        n = np.array([])
        r = rp.asarray([])
        assert_eq(rp.nancumprod(r), np.nancumprod(n))


# === Edge cases ===


class TestNanExtensionEdgeCases:
    """Test edge cases shared across NaN extension functions."""

    def test_3d_arrays(self):
        """Test all functions on 3D arrays."""
        n_data = [[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [np.nan, 8.0]]]
        r_data = [[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [float('nan'), 8.0]]]
        n = np.array(n_data)
        r = rp.asarray(r_data)

        # nanmedian
        assert rp.nanmedian(r) == pytest.approx(np.nanmedian(n))

        # nanpercentile (single q returns array in rumpy)
        assert abs(float(rp.nanpercentile(r, 50)[0]) - np.nanpercentile(n, 50)) < 1e-10

        # nanquantile (single q returns array in rumpy)
        assert abs(float(rp.nanquantile(r, 0.5)[0]) - np.nanquantile(n, 0.5)) < 1e-10

        # nancumsum
        assert_eq(rp.nancumsum(r), np.nancumsum(n))

        # nancumprod
        assert_eq(rp.nancumprod(r), np.nancumprod(n))

    def test_negative_values(self):
        """Test with negative values and NaN."""
        n = np.array([-3.0, np.nan, -1.0, 0.0, 1.0, np.nan, 3.0])
        r = rp.asarray([-3.0, float('nan'), -1.0, 0.0, 1.0, float('nan'), 3.0])

        assert rp.nanmedian(r) == pytest.approx(np.nanmedian(n))
        assert_eq(rp.nancumsum(r), np.nancumsum(n))

    def test_inf_values(self):
        """Test with infinity values and NaN (inf should not be ignored)."""
        n = np.array([1.0, np.inf, np.nan, 3.0])
        r = rp.asarray([1.0, float('inf'), float('nan'), 3.0])

        # nancumsum should keep inf
        assert_eq(rp.nancumsum(r), np.nancumsum(n))
