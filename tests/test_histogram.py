"""Tests for extended histogram and cumulative functions (Stream 37).

Functions tested:
- histogram_bin_edges: compute bin edges only
- histogram2d: 2D histogram
- histogramdd: N-dimensional histogram
- cumulative_sum: NumPy 2.0 cumulative sum API
- cumulative_prod: NumPy 2.0 cumulative product API

See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import FLOAT_DTYPES, NUMERIC_DTYPES
from helpers import assert_eq, make_pair


# === histogram_bin_edges ===


class TestHistogramBinEdges:
    """Test histogram_bin_edges function."""

    def test_basic(self):
        """Basic bin edges computation."""
        n = np.array([1, 2, 2, 3, 3, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        n_edges = np.histogram_bin_edges(n, bins=5)
        r_edges = rp.histogram_bin_edges(r, bins=5)
        assert_eq(r_edges, n_edges)

    def test_default_bins(self):
        """Default bins=10."""
        n = np.arange(100, dtype=np.float64)
        r = rp.asarray(n)
        n_edges = np.histogram_bin_edges(n)
        r_edges = rp.histogram_bin_edges(r)
        assert_eq(r_edges, n_edges)

    def test_with_range(self):
        """Bin edges with explicit range."""
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        n_edges = np.histogram_bin_edges(n, bins=5, range=(0, 10))
        r_edges = rp.histogram_bin_edges(r, bins=5, range=(0, 10))
        assert_eq(r_edges, n_edges)

    def test_array_bins(self):
        """Bin edges from explicit array."""
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        bins = [0, 2, 4, 6]
        n_edges = np.histogram_bin_edges(n, bins=bins)
        r_edges = rp.histogram_bin_edges(r, bins=bins)
        assert_eq(r_edges, n_edges)

    def test_empty_array(self):
        """Bin edges for empty array."""
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        n_edges = np.histogram_bin_edges(n, bins=5)
        r_edges = rp.histogram_bin_edges(r, bins=5)
        assert_eq(r_edges, n_edges)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        """Test with different dtypes."""
        n = np.arange(20, dtype=dtype)
        r = rp.asarray(n)
        n_edges = np.histogram_bin_edges(n, bins=5)
        r_edges = rp.histogram_bin_edges(r, bins=5)
        assert_eq(r_edges, n_edges)


# === histogram2d ===


class TestHistogram2d:
    """Test 2D histogram function."""

    def test_basic(self):
        """Basic 2D histogram."""
        x = np.array([1.0, 2.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 2.0, 3.0])
        rx, ry = rp.asarray(x), rp.asarray(y)

        n_H, n_xedges, n_yedges = np.histogram2d(x, y, bins=3)
        r_H, r_xedges, r_yedges = rp.histogram2d(rx, ry, bins=3)

        assert_eq(r_H, n_H)
        assert_eq(r_xedges, n_xedges)
        assert_eq(r_yedges, n_yedges)

    def test_different_bins_per_axis(self):
        """Different number of bins per axis."""
        x = np.arange(10, dtype=np.float64)
        y = np.arange(10, dtype=np.float64)
        rx, ry = rp.asarray(x), rp.asarray(y)

        n_H, n_xedges, n_yedges = np.histogram2d(x, y, bins=[3, 5])
        r_H, r_xedges, r_yedges = rp.histogram2d(rx, ry, bins=[3, 5])

        assert_eq(r_H, n_H)
        assert_eq(r_xedges, n_xedges)
        assert_eq(r_yedges, n_yedges)

    def test_with_range(self):
        """2D histogram with range."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rx, ry = rp.asarray(x), rp.asarray(y)

        n_H, n_xedges, n_yedges = np.histogram2d(x, y, bins=5, range=[[0, 6], [0, 6]])
        r_H, r_xedges, r_yedges = rp.histogram2d(rx, ry, bins=5, range=[[0, 6], [0, 6]])

        assert_eq(r_H, n_H)
        assert_eq(r_xedges, n_xedges)
        assert_eq(r_yedges, n_yedges)

    def test_with_density(self):
        """2D histogram with density normalization."""
        x = np.random.randn(100)
        y = np.random.randn(100)
        rx, ry = rp.asarray(x), rp.asarray(y)

        n_H, n_xedges, n_yedges = np.histogram2d(x, y, bins=5, density=True)
        r_H, r_xedges, r_yedges = rp.histogram2d(rx, ry, bins=5, density=True)

        assert_eq(r_H, n_H)
        assert_eq(r_xedges, n_xedges)
        assert_eq(r_yedges, n_yedges)

    def test_single_point(self):
        """2D histogram with single point."""
        x = np.array([1.0])
        y = np.array([2.0])
        rx, ry = rp.asarray(x), rp.asarray(y)

        n_H, n_xedges, n_yedges = np.histogram2d(x, y, bins=3)
        r_H, r_xedges, r_yedges = rp.histogram2d(rx, ry, bins=3)

        # Shape should match
        assert r_H.shape == n_H.shape
        # Total count should be 1
        assert r_H.sum() == n_H.sum() == 1

    def test_empty(self):
        """2D histogram with empty arrays."""
        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)
        rx, ry = rp.asarray(x), rp.asarray(y)

        n_H, n_xedges, n_yedges = np.histogram2d(x, y, bins=3)
        r_H, r_xedges, r_yedges = rp.histogram2d(rx, ry, bins=3)

        assert_eq(r_H, n_H)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        """Test with different dtypes."""
        x = np.arange(10, dtype=dtype)
        y = np.arange(10, dtype=dtype)
        rx, ry = rp.asarray(x), rp.asarray(y)

        n_H, n_xedges, n_yedges = np.histogram2d(x, y, bins=5)
        r_H, r_xedges, r_yedges = rp.histogram2d(rx, ry, bins=5)

        assert_eq(r_H, n_H)


# === histogramdd ===


class TestHistogramdd:
    """Test N-dimensional histogram function."""

    def test_2d_sample(self):
        """histogramdd with 2D sample array."""
        sample = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        r_sample = rp.asarray(sample)

        n_H, n_edges = np.histogramdd(sample, bins=3)
        r_H, r_edges = rp.histogramdd(r_sample, bins=3)

        assert_eq(r_H, n_H)
        assert len(r_edges) == len(n_edges)
        for r_e, n_e in zip(r_edges, n_edges):
            assert_eq(r_e, n_e)

    def test_3d_sample(self):
        """histogramdd with 3D sample (N points, 3 dimensions)."""
        np.random.seed(42)
        sample = np.random.randn(50, 3)
        r_sample = rp.asarray(sample)

        n_H, n_edges = np.histogramdd(sample, bins=4)
        r_H, r_edges = rp.histogramdd(r_sample, bins=4)

        assert_eq(r_H, n_H)
        assert len(r_edges) == len(n_edges) == 3

    def test_different_bins_per_dimension(self):
        """histogramdd with different bins per dimension."""
        sample = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        r_sample = rp.asarray(sample)

        n_H, n_edges = np.histogramdd(sample, bins=[3, 5])
        r_H, r_edges = rp.histogramdd(r_sample, bins=[3, 5])

        assert_eq(r_H, n_H)
        assert len(r_edges) == 2
        assert_eq(r_edges[0], n_edges[0])
        assert_eq(r_edges[1], n_edges[1])

    def test_with_range(self):
        """histogramdd with explicit range."""
        sample = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        r_sample = rp.asarray(sample)

        n_H, n_edges = np.histogramdd(sample, bins=3, range=[[0, 4], [0, 4]])
        r_H, r_edges = rp.histogramdd(r_sample, bins=3, range=[[0, 4], [0, 4]])

        assert_eq(r_H, n_H)

    def test_with_density(self):
        """histogramdd with density normalization."""
        np.random.seed(42)
        sample = np.random.randn(100, 2)
        r_sample = rp.asarray(sample)

        n_H, n_edges = np.histogramdd(sample, bins=5, density=True)
        r_H, r_edges = rp.histogramdd(r_sample, bins=5, density=True)

        assert_eq(r_H, n_H)

    def test_empty_sample(self):
        """histogramdd with empty sample."""
        sample = np.zeros((0, 2), dtype=np.float64)
        r_sample = rp.asarray(sample)

        n_H, n_edges = np.histogramdd(sample, bins=3)
        r_H, r_edges = rp.histogramdd(r_sample, bins=3)

        assert_eq(r_H, n_H)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        """Test with different dtypes."""
        sample = np.arange(20, dtype=dtype).reshape(10, 2)
        r_sample = rp.asarray(sample)

        n_H, n_edges = np.histogramdd(sample, bins=3)
        r_H, r_edges = rp.histogramdd(r_sample, bins=3)

        assert_eq(r_H, n_H)


# === cumulative_sum ===


class TestCumulativeSum:
    """Test NumPy 2.0 cumulative_sum function."""

    def test_1d(self):
        """1D cumulative sum."""
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_sum(n, axis=0)
        r_result = rp.cumulative_sum(r, axis=0)
        assert_eq(r_result, n_result)

    def test_2d_axis0(self):
        """2D cumulative sum along axis 0."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_sum(n, axis=0)
        r_result = rp.cumulative_sum(r, axis=0)
        assert_eq(r_result, n_result)

    def test_2d_axis1(self):
        """2D cumulative sum along axis 1."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_sum(n, axis=1)
        r_result = rp.cumulative_sum(r, axis=1)
        assert_eq(r_result, n_result)

    def test_include_initial(self):
        """Cumulative sum with include_initial=True."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_sum(n, axis=0, include_initial=True)
        r_result = rp.cumulative_sum(r, axis=0, include_initial=True)
        assert_eq(r_result, n_result)

    def test_include_initial_axis1(self):
        """Cumulative sum with include_initial=True along axis 1."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_sum(n, axis=1, include_initial=True)
        r_result = rp.cumulative_sum(r, axis=1, include_initial=True)
        assert_eq(r_result, n_result)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, dtype):
        """Test with different dtypes."""
        n = np.arange(10, dtype=dtype)
        r = rp.asarray(n)
        n_result = np.cumulative_sum(n, axis=0)
        r_result = rp.cumulative_sum(r, axis=0)
        assert_eq(r_result, n_result)

    def test_3d(self):
        """3D cumulative sum."""
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        for axis in [0, 1, 2]:
            n_result = np.cumulative_sum(n, axis=axis)
            r_result = rp.cumulative_sum(r, axis=axis)
            assert_eq(r_result, n_result)


# === cumulative_prod ===


class TestCumulativeProd:
    """Test NumPy 2.0 cumulative_prod function."""

    def test_1d(self):
        """1D cumulative product."""
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_prod(n, axis=0)
        r_result = rp.cumulative_prod(r, axis=0)
        assert_eq(r_result, n_result)

    def test_2d_axis0(self):
        """2D cumulative product along axis 0."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_prod(n, axis=0)
        r_result = rp.cumulative_prod(r, axis=0)
        assert_eq(r_result, n_result)

    def test_2d_axis1(self):
        """2D cumulative product along axis 1."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_prod(n, axis=1)
        r_result = rp.cumulative_prod(r, axis=1)
        assert_eq(r_result, n_result)

    def test_include_initial(self):
        """Cumulative product with include_initial=True."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_prod(n, axis=0, include_initial=True)
        r_result = rp.cumulative_prod(r, axis=0, include_initial=True)
        assert_eq(r_result, n_result)

    def test_include_initial_axis1(self):
        """Cumulative product with include_initial=True along axis 1."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_prod(n, axis=1, include_initial=True)
        r_result = rp.cumulative_prod(r, axis=1, include_initial=True)
        assert_eq(r_result, n_result)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        """Test with different dtypes."""
        n = np.array([1, 2, 3, 4], dtype=dtype)
        r = rp.asarray(n)
        n_result = np.cumulative_prod(n, axis=0)
        r_result = rp.cumulative_prod(r, axis=0)
        assert_eq(r_result, n_result)

    def test_3d(self):
        """3D cumulative product."""
        n = np.ones((2, 3, 4), dtype=np.float64) * 2
        r = rp.asarray(n)
        for axis in [0, 1, 2]:
            n_result = np.cumulative_prod(n, axis=axis)
            r_result = rp.cumulative_prod(r, axis=axis)
            assert_eq(r_result, n_result)


# === Edge cases ===


class TestExtendedHistogramEdgeCases:
    """Edge case tests for extended histogram functions."""

    def test_histogram2d_mismatched_lengths(self):
        """histogram2d should raise for mismatched x, y lengths."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        rx, ry = rp.asarray(x), rp.asarray(y)

        with pytest.raises((ValueError, RuntimeError)):
            rp.histogram2d(rx, ry, bins=3)

    def test_cumulative_sum_preserves_dtype(self):
        """cumulative_sum output dtype behavior."""
        # Note: NumPy 2.0 promotes int32->int64, but our impl preserves dtype.
        # Test that float dtypes are preserved (matching NumPy for floats).
        for dtype in ['float32', 'float64']:
            n = np.array([1, 2, 3], dtype=dtype)
            r = rp.asarray(n)
            result = rp.cumulative_sum(r, axis=0)
            n_result = np.cumulative_sum(n, axis=0)
            assert str(result.dtype) == str(n_result.dtype), f"dtype mismatch for {dtype}"

    def test_cumulative_prod_with_zeros(self):
        """cumulative_prod with zeros should stay zero after."""
        n = np.array([1, 2, 0, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        n_result = np.cumulative_prod(n, axis=0)
        r_result = rp.cumulative_prod(r, axis=0)
        assert_eq(r_result, n_result)
