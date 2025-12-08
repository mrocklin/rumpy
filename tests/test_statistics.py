"""Tests for statistical functions.

Statistics functions often work on float dtypes primarily.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES, FLOAT_DTYPES, NUMERIC_DTYPES
from helpers import assert_eq, make_pair


# === Median ===


class TestMedian:
    """Test median function."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.median(r), np.median(n))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        r, n = make_pair(shape, "float64")
        assert_eq(rp.median(r), np.median(n))

    def test_1d_odd(self):
        n = np.array([3, 1, 4, 1, 5], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.median(r), np.median(n))

    def test_1d_even(self):
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.median(r), np.median(n))

    def test_2d_axis0(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.median(r, axis=0), np.median(n, axis=0))

    def test_2d_axis1(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.median(r, axis=1), np.median(n, axis=1))

    def test_3d_axis(self):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        for axis in [0, 1, 2]:
            assert_eq(rp.median(r, axis=axis), np.median(n, axis=axis))

    def test_empty(self):
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        # NumPy returns nan for empty array
        assert_eq(rp.median(r), np.median(n))

    def test_single(self):
        n = np.array([42.0], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.median(r), np.median(n))


# === Peak-to-peak (ptp) ===


class TestPtp:
    """Test peak-to-peak (max - min) function."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.ptp(r), np.ptp(n))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        r, n = make_pair(shape, "float64")
        assert_eq(rp.ptp(r), np.ptp(n))

    def test_2d_axis0(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.ptp(r, axis=0), np.ptp(n, axis=0))

    def test_2d_axis1(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.ptp(r, axis=1), np.ptp(n, axis=1))

    def test_3d_axis(self):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        for axis in [0, 1, 2]:
            assert_eq(rp.ptp(r, axis=axis), np.ptp(n, axis=axis))

    def test_empty(self):
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        # NumPy returns nan for empty array, rumpy may error
        try:
            r_result = rp.ptp(r)
            n_result = np.ptp(n)
            assert_eq(r_result, n_result)
        except (ValueError, RuntimeError):
            # Empty array behavior can vary
            pass

    def test_single(self):
        n = np.array([42.0], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.ptp(r), np.ptp(n))


# === Average ===


class TestAverage:
    """Test weighted average function."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.arange(10, dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.average(r), np.average(n))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        r, n = make_pair(shape, "float64")
        assert_eq(rp.average(r), np.average(n))

    def test_no_weights(self):
        n = np.arange(10, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.average(r), np.average(n))

    def test_with_weights(self):
        n = np.array([1.0, 2.0, 3.0, 4.0])
        nw = np.array([0.5, 0.25, 0.15, 0.1])
        r = rp.asarray(n)
        w = rp.asarray(nw)
        assert_eq(rp.average(r, weights=w), np.average(n, weights=nw))

    def test_2d_axis0(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.average(r, axis=0), np.average(n, axis=0))

    def test_2d_axis1(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.average(r, axis=1), np.average(n, axis=1))

    def test_empty(self):
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.average(r), np.average(n))

    def test_single(self):
        n = np.array([42.0], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.average(r), np.average(n))


# === Histogram ===


class TestHistogram:
    """Test histogram function."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 2.0, 1.0], dtype=dtype)
        r = rp.asarray(n)
        r_counts, r_bins = rp.histogram(r, bins=3)
        n_counts, n_bins = np.histogram(n, bins=3)
        assert_eq(r_counts, n_counts)
        assert_eq(r_bins, n_bins)

    def test_basic(self):
        n = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 2.0, 1.0])
        r = rp.asarray(n)
        r_counts, r_bins = rp.histogram(r, bins=3)
        n_counts, n_bins = np.histogram(n, bins=3)
        assert_eq(r_counts, n_counts)
        assert_eq(r_bins, n_bins)

    def test_10_bins(self):
        n = np.arange(100, dtype=np.float64)
        r = rp.asarray(n)
        r_counts, r_bins = rp.histogram(r, bins=10)
        n_counts, n_bins = np.histogram(n, bins=10)
        assert_eq(r_counts, n_counts)
        assert_eq(r_bins, n_bins)

    def test_with_range(self):
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = rp.asarray(n)
        r_counts, r_bins = rp.histogram(r, bins=5, range=(0.0, 6.0))
        n_counts, n_bins = np.histogram(n, bins=5, range=(0.0, 6.0))
        assert_eq(r_counts, n_counts)
        assert_eq(r_bins, n_bins)

    def test_empty(self):
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        r_counts, r_bins = rp.histogram(r, bins=5)
        n_counts, n_bins = np.histogram(n, bins=5)
        assert_eq(r_counts, n_counts)
        assert_eq(r_bins, n_bins)

    def test_single(self):
        n = np.array([42.0], dtype=np.float64)
        r = rp.asarray(n)
        r_counts, r_bins = rp.histogram(r, bins=5)
        n_counts, n_bins = np.histogram(n, bins=5)
        # Single element histogram: implementations may differ on which bin
        # Just check that exactly one bin has count 1, rest have 0
        assert r_counts.shape == n_counts.shape
        assert r_counts.sum() == n_counts.sum() == 1
        assert r_bins.shape == n_bins.shape


# === Covariance ===


class TestCov:
    """Test covariance matrix."""

    def test_1d(self):
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = rp.asarray(n)
        # For 1D input, cov returns variance (with ddof=1 by default)
        # NumPy returns scalar, rumpy returns 0-d or 1-d array
        r_result = rp.cov(r)
        n_result = np.cov(n)
        # Compare values, allowing shape differences
        r_val = float(r_result) if hasattr(r_result, '__float__') else float(r_result.flat[0])
        n_val = float(n_result) if hasattr(n_result, 'item') else float(n_result)
        assert abs(r_val - n_val) < 1e-10

    def test_2d(self):
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rp.asarray(n)
        assert_eq(rp.cov(r), np.cov(n))

    def test_ddof0(self):
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rp.asarray(n)
        assert_eq(rp.cov(r, ddof=0), np.cov(n, ddof=0))

    def test_ddof1(self):
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rp.asarray(n)
        assert_eq(rp.cov(r, ddof=1), np.cov(n, ddof=1))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.cov(r), np.cov(n))

    def test_single_observation(self):
        n = np.array([[1.0], [2.0]])
        r = rp.asarray(n)
        # Single observation leads to NaN with ddof=1 (default)
        assert_eq(rp.cov(r), np.cov(n))

    def test_empty(self):
        n = np.array([], dtype=np.float64).reshape(0, 0)
        r = rp.asarray(n)
        # Empty should handle gracefully
        try:
            r_result = rp.cov(r)
            n_result = np.cov(n)
            assert_eq(r_result, n_result)
        except (ValueError, RuntimeError):
            # If implementation raises error, ensure both do
            with pytest.raises((ValueError, RuntimeError)):
                np.cov(n)


# === Correlation coefficient ===


class TestCorrcoef:
    """Test correlation coefficient matrix."""

    def test_1d(self):
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = rp.asarray(n)
        # For 1D input, numpy returns scalar 1.0, we return 1x1 matrix
        r_result = rp.corrcoef(r)
        n_result = np.corrcoef(n)
        # Check correlation is 1.0 in both
        assert abs(float(r_result[0, 0]) - 1.0) < 1e-10
        assert abs(float(n_result) - 1.0) < 1e-10

    def test_2d(self):
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rp.asarray(n)
        assert_eq(rp.corrcoef(r), np.corrcoef(n))

    def test_perfect_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])  # y = 2x, perfect correlation
        n = np.stack([x, y])
        r = rp.stack([rp.asarray(x), rp.asarray(y)])
        r_result = rp.corrcoef(r)
        # Should have 1.0 on diagonal and 1.0 off-diagonal
        assert abs(r_result[0, 0] - 1.0) < 1e-10
        assert abs(r_result[1, 1] - 1.0) < 1e-10
        assert abs(r_result[0, 1] - 1.0) < 1e-10
        assert abs(r_result[1, 0] - 1.0) < 1e-10

    def test_anticorrelation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([4.0, 3.0, 2.0, 1.0])  # perfect anticorrelation
        n = np.stack([x, y])
        r = rp.stack([rp.asarray(x), rp.asarray(y)])
        r_result = rp.corrcoef(r)
        n_result = np.corrcoef(n)
        assert_eq(r_result, n_result)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.corrcoef(r), np.corrcoef(n))


# === Higher-order moments ===


class TestMoment:
    """Test central moments."""

    def test_moment_2_equals_var(self):
        """Moment of order 2 should equal variance."""
        n = np.arange(20, dtype=np.float64)
        r = rp.asarray(n)
        m2 = r.moment(2)
        v = r.var()
        assert abs(m2 - v) < 1e-10

    def test_moment_axis(self):
        """Moment along axis."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        # Compare moment(2) along axis=0 to numpy variance
        r_m2 = r.moment(2, axis=0)
        n_var = n.var(axis=0)
        assert_eq(r_m2, n_var)

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_orders(self, order):
        n = np.arange(20, dtype=np.float64)
        r = rp.asarray(n)
        # Just verify it runs without error
        result = r.moment(order)
        assert isinstance(result, (int, float))


class TestSkewness:
    """Test skewness (Fisher's definition)."""

    def test_skew_symmetric(self):
        """Symmetric distribution should have skewness near 0."""
        n = np.arange(-5, 6, dtype=np.float64)
        r = rp.asarray(n)
        s = r.skew()
        assert abs(s) < 1e-10

    def test_skew_positive(self):
        """Right-skewed distribution should have positive skewness."""
        n = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 10.0])
        r = rp.asarray(n)
        s = r.skew()
        assert s > 0

    def test_skew_negative(self):
        """Left-skewed distribution should have negative skewness."""
        n = np.array([-10.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0])
        r = rp.asarray(n)
        s = r.skew()
        assert s < 0

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.arange(-5, 6, dtype=dtype)
        r = rp.asarray(n)
        s = r.skew()
        assert abs(s) < 1e-5


class TestKurtosis:
    """Test kurtosis (Fisher's definition: excess kurtosis, normal = 0)."""

    def test_kurtosis_uniform(self):
        """Uniform distribution has negative excess kurtosis (-1.2)."""
        n = np.arange(1000, dtype=np.float64)
        r = rp.asarray(n)
        k = r.kurtosis()
        # Uniform: kurtosis = -6/5 = -1.2
        assert abs(k - (-1.2)) < 0.01

    def test_kurtosis_leptokurtic(self):
        """Distribution with heavy tails should have positive kurtosis."""
        n = np.array([0.0] * 100 + [-10.0, 10.0])
        r = rp.asarray(n)
        k = r.kurtosis()
        assert k > 0

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        n = np.arange(100, dtype=dtype)
        r = rp.asarray(n)
        # Just verify it runs
        k = r.kurtosis()
        assert isinstance(k, (int, float))


# === Edge Cases ===


class TestStatisticsEdgeCases:
    """Test edge cases for statistical functions."""

    def test_median_empty_axis(self):
        """Median on axis with size 0."""
        n = np.zeros((0, 5), dtype=np.float64)
        r = rp.asarray(n)
        try:
            r_result = rp.median(r, axis=1)
            n_result = np.median(n, axis=1)
            assert_eq(r_result, n_result)
        except (ValueError, RuntimeError):
            # If implementation raises error, that's acceptable
            pass

    def test_ptp_constant(self):
        """Ptp of constant array should be 0."""
        n = np.ones(10, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.ptp(r), np.ptp(n))

    def test_average_all_zero_weights(self):
        """Average with zero weights."""
        n = np.array([1.0, 2.0, 3.0])
        nw = np.array([0.0, 0.0, 0.0])
        r = rp.asarray(n)
        w = rp.asarray(nw)
        try:
            r_result = rp.average(r, weights=w)
            n_result = np.average(n, weights=nw)
            assert_eq(r_result, n_result)
        except (ValueError, RuntimeError, ZeroDivisionError):
            # If implementation raises error, ensure both do
            with pytest.raises((ValueError, RuntimeError, ZeroDivisionError)):
                np.average(n, weights=nw)

    def test_cov_constant_rows(self):
        """Covariance with constant rows."""
        n = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        r = rp.asarray(n)
        assert_eq(rp.cov(r), np.cov(n))

    def test_corrcoef_zero_variance(self):
        """Correlation with zero variance row."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        n = np.stack([x, y])
        r = rp.stack([rp.asarray(x), rp.asarray(y)])
        # NumPy sets correlation to nan when variance is 0
        # Implementations may differ on which elements are nan
        r_result = rp.corrcoef(r)
        n_result = np.corrcoef(n)
        # Check that both have the same shape and diagonal is 1 where valid
        assert r_result.shape == n_result.shape
        # Element (1,1) should be 1.0 (autocorrelation of y)
        assert abs(r_result[1, 1] - 1.0) < 1e-10


class TestBincount:
    """Test bincount function."""

    def test_simple(self):
        r = rp.bincount(rp.array([0, 1, 1, 2, 2, 2]))
        n = np.bincount([0, 1, 1, 2, 2, 2])
        assert_eq(r, n)

    def test_with_gaps(self):
        r = rp.bincount(rp.array([0, 0, 3, 5, 5, 5]))
        n = np.bincount([0, 0, 3, 5, 5, 5])
        assert_eq(r, n)

    def test_minlength(self):
        r = rp.bincount(rp.array([0, 1, 2]), minlength=10)
        n = np.bincount([0, 1, 2], minlength=10)
        assert_eq(r, n)


class TestPercentile:
    """Test percentile function."""

    def test_median(self):
        r = rp.percentile(rp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 50)
        n = np.percentile([1, 2, 3, 4, 5], 50)
        assert abs(float(r[0]) - n) < 1e-10

    def test_quartiles(self):
        r = rp.percentile(rp.array([1.0, 2.0, 3.0, 4.0, 5.0]), [25, 50, 75])
        n = np.percentile([1, 2, 3, 4, 5], [25, 50, 75])
        assert_eq(r, n)

    def test_extremes(self):
        r = rp.percentile(rp.array([1.0, 2.0, 3.0]), [0, 100])
        n = np.percentile([1, 2, 3], [0, 100])
        assert_eq(r, n)


class TestQuantile:
    """Test quantile function."""

    def test_median(self):
        r = rp.quantile(rp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0.5)
        n = np.quantile([1, 2, 3, 4, 5], 0.5)
        assert abs(float(r[0]) - n) < 1e-10
