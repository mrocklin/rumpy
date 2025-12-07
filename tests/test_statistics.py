"""Tests for statistical functions."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestVar:
    """Test variance."""

    def test_var_1d(self):
        arr = rp.arange(10)
        r = arr.var()
        n = np.arange(10).var()
        assert abs(r - n) < 1e-10

    def test_var_2d(self):
        arr = rp.arange(12).reshape(3, 4)
        r = arr.var()
        n = np.arange(12).reshape(3, 4).var()
        assert abs(r - n) < 1e-10

    def test_var_axis0(self):
        arr = rp.arange(12, dtype="float64").reshape(3, 4)
        r = arr.var(axis=0)
        n = np.arange(12, dtype=np.float64).reshape(3, 4).var(axis=0)
        assert_eq(r, n)

    def test_var_axis1(self):
        arr = rp.arange(12, dtype="float64").reshape(3, 4)
        r = arr.var(axis=1)
        n = np.arange(12, dtype=np.float64).reshape(3, 4).var(axis=1)
        assert_eq(r, n)


class TestStd:
    """Test standard deviation."""

    def test_std_1d(self):
        arr = rp.arange(10, dtype="float64")
        r = arr.std()
        n = np.arange(10, dtype=np.float64).std()
        assert abs(r - n) < 1e-10

    def test_std_2d(self):
        arr = rp.arange(12, dtype="float64").reshape(3, 4)
        r = arr.std()
        n = np.arange(12, dtype=np.float64).reshape(3, 4).std()
        assert abs(r - n) < 1e-10

    def test_std_axis0(self):
        arr = rp.arange(12, dtype="float64").reshape(3, 4)
        r = arr.std(axis=0)
        n = np.arange(12, dtype=np.float64).reshape(3, 4).std(axis=0)
        assert_eq(r, n)


class TestMedian:
    """Test median function."""

    def test_median_1d(self):
        r = rp.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float64)
        assert rp.median(r) == pytest.approx(np.median(n))

    def test_median_1d_odd(self):
        r = rp.asarray([3, 1, 4, 1, 5])
        n = np.array([3, 1, 4, 1, 5], dtype=np.float64)
        assert rp.median(r) == pytest.approx(np.median(n))

    def test_median_2d_flat(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert rp.median(r) == pytest.approx(np.median(n))

    def test_median_axis0(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(rp.median(r, axis=0), np.median(n, axis=0))

    def test_median_axis1(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(rp.median(r, axis=1), np.median(n, axis=1))


class TestAverage:
    """Test weighted average function."""

    def test_average_no_weights(self):
        r = rp.arange(10, dtype="float64")
        n = np.arange(10, dtype=np.float64)
        assert rp.average(r) == pytest.approx(np.average(n))

    def test_average_with_weights(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0])
        w = rp.asarray([0.5, 0.25, 0.15, 0.1])
        n = np.array([1.0, 2.0, 3.0, 4.0])
        nw = np.array([0.5, 0.25, 0.15, 0.1])
        assert rp.average(r, weights=w) == pytest.approx(np.average(n, weights=nw))

    def test_average_2d_no_axis(self):
        r = rp.arange(12, dtype="float64").reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert rp.average(r) == pytest.approx(np.average(n))

    def test_average_2d_axis0(self):
        r = rp.arange(12, dtype="float64").reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(rp.average(r, axis=0), np.average(n, axis=0))

    def test_average_2d_axis1(self):
        r = rp.arange(12, dtype="float64").reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(rp.average(r, axis=1), np.average(n, axis=1))


class TestPtp:
    """Test peak-to-peak (max - min) function."""

    def test_ptp_1d(self):
        r = rp.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float64)
        assert rp.ptp(r) == pytest.approx(np.ptp(n))

    def test_ptp_2d_flat(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert rp.ptp(r) == pytest.approx(np.ptp(n))

    def test_ptp_axis0(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(rp.ptp(r, axis=0), np.ptp(n, axis=0))

    def test_ptp_axis1(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(rp.ptp(r, axis=1), np.ptp(n, axis=1))


class TestHistogram:
    """Test histogram function."""

    def test_histogram_basic(self):
        r = rp.asarray([1.0, 2.0, 1.0, 3.0, 2.0, 2.0, 1.0])
        n = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 2.0, 1.0])
        r_counts, r_bins = rp.histogram(r, bins=3)
        n_counts, n_bins = np.histogram(n, bins=3)
        assert_eq(r_counts, n_counts)
        assert_eq(r_bins, n_bins)

    def test_histogram_10_bins(self):
        r = rp.arange(100)
        n = np.arange(100, dtype=np.float64)
        r_counts, r_bins = rp.histogram(r, bins=10)
        n_counts, n_bins = np.histogram(n, bins=10)
        assert_eq(r_counts, n_counts)
        assert_eq(r_bins, n_bins)

    def test_histogram_with_range(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r_counts, r_bins = rp.histogram(r, bins=5, range=(0.0, 6.0))
        n_counts, n_bins = np.histogram(n, bins=5, range=(0.0, 6.0))
        assert_eq(r_counts, n_counts)
        assert_eq(r_bins, n_bins)


class TestCorrcoef:
    """Test correlation coefficient matrix."""

    def test_corrcoef_1d(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # For 1D input, numpy returns scalar 1.0, we return 1x1 matrix
        # Both are valid - we just check value is 1.0
        r_result = rp.corrcoef(r)
        n_result = np.corrcoef(n)
        assert abs(float(r_result[0, 0]) - 1.0) < 1e-10
        assert abs(float(n_result) - 1.0) < 1e-10

    def test_corrcoef_2d(self):
        r = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r_result = rp.corrcoef(r)
        n_result = np.corrcoef(n)
        assert_eq(r_result, n_result)

    def test_corrcoef_perfect_correlation(self):
        x = rp.asarray([1.0, 2.0, 3.0, 4.0])
        y = rp.asarray([2.0, 4.0, 6.0, 8.0])  # y = 2x, perfect correlation
        r = rp.corrcoef(rp.stack([x, y]))
        # Should have 1.0 on diagonal and 1.0 off-diagonal (perfect correlation)
        assert abs(r[0, 0] - 1.0) < 1e-10
        assert abs(r[1, 1] - 1.0) < 1e-10
        assert abs(r[0, 1] - 1.0) < 1e-10
        assert abs(r[1, 0] - 1.0) < 1e-10


class TestCov:
    """Test covariance matrix."""

    def test_cov_1d(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # For 1D input, cov returns variance (with ddof=1 by default)
        r_result = rp.cov(r)
        n_result = np.cov(n)
        # NumPy returns scalar for 1D, we might return 0-d array
        if hasattr(r_result, "__float__"):
            assert float(r_result) == pytest.approx(float(n_result))
        else:
            assert r_result == pytest.approx(n_result)

    def test_cov_2d(self):
        r = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r_result = rp.cov(r)
        n_result = np.cov(n)
        assert_eq(r_result, n_result)

    def test_cov_ddof0(self):
        r = rp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r_result = rp.cov(r, ddof=0)
        n_result = np.cov(n, ddof=0)
        assert_eq(r_result, n_result)


class TestArgmax:
    """Test argmax."""

    def test_argmax_1d(self):
        arr = rp.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        r = arr.argmax()
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6]).argmax()
        assert r == n

    def test_argmax_2d(self):
        arr = rp.arange(12).reshape(3, 4)
        r = arr.argmax()
        n = np.arange(12).reshape(3, 4).argmax()
        assert r == n


class TestArgmin:
    """Test argmin."""

    def test_argmin_1d(self):
        arr = rp.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        r = arr.argmin()
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6]).argmin()
        assert r == n

    def test_argmin_2d(self):
        arr = rp.arange(12).reshape(3, 4)
        r = arr.argmin()
        n = np.arange(12).reshape(3, 4).argmin()
        assert r == n


class TestSort:
    """Test sort."""

    def test_sort_1d(self):
        arr = rp.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        r = rp.sort(arr)
        n = np.sort(np.array([3, 1, 4, 1, 5, 9, 2, 6]))
        assert_eq(r, n)

    def test_sort_2d_flattened(self):
        arr = rp.asarray([[3, 1], [4, 2]])
        r = rp.sort(arr, axis=None)
        n = np.sort(np.array([[3, 1], [4, 2]]), axis=None)
        assert_eq(r, n)


class TestArgsort:
    """Test argsort."""

    def test_argsort_1d(self):
        arr = rp.asarray([3, 1, 4, 1, 5])
        r = rp.argsort(arr)
        n = np.argsort(np.array([3, 1, 4, 1, 5]))
        assert_eq(r, n)


class TestMoment:
    """Test central moments."""

    def test_moment_2_equals_var(self):
        """Moment of order 2 should equal variance."""
        arr = rp.arange(20, dtype="float64")
        m2 = arr.moment(2)
        v = arr.var()
        assert abs(m2 - v) < 1e-10

    def test_moment_axis(self):
        """Moment along axis."""
        arr = rp.arange(12, dtype="float64").reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        # Compare moment(2) along axis=0 to numpy variance
        r_m2 = arr.moment(2, axis=0)
        n_var = n.var(axis=0)
        assert_eq(r_m2, n_var)


class TestSkewness:
    """Test skewness (Fisher's definition)."""

    def test_skew_symmetric(self):
        """Symmetric distribution should have skewness near 0."""
        # Symmetric: -5, -4, ..., 4, 5
        arr = rp.arange(-5, 6, dtype="float64")
        s = arr.skew()
        assert abs(s) < 1e-10

    def test_skew_positive(self):
        """Right-skewed distribution should have positive skewness."""
        # Heavy right tail
        arr = rp.asarray([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 10.0])
        s = arr.skew()
        assert s > 0

    def test_skew_negative(self):
        """Left-skewed distribution should have negative skewness."""
        # Heavy left tail
        arr = rp.asarray([-10.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0])
        s = arr.skew()
        assert s < 0


class TestKurtosis:
    """Test kurtosis (Fisher's definition: excess kurtosis, normal = 0)."""

    def test_kurtosis_uniform(self):
        """Uniform distribution has negative excess kurtosis (-1.2)."""
        arr = rp.arange(1000, dtype="float64")
        k = arr.kurtosis()
        # Uniform: kurtosis = -6/5 = -1.2
        assert abs(k - (-1.2)) < 0.01

    def test_kurtosis_leptokurtic(self):
        """Distribution with heavy tails should have positive kurtosis."""
        # Outliers create heavy tails
        arr = rp.asarray([0.0] * 100 + [-10.0, 10.0])
        k = arr.kurtosis()
        assert k > 0


class TestUnique:
    """Test unique."""

    def test_unique_1d(self):
        arr = rp.asarray([3, 1, 4, 1, 5, 9, 2, 6, 5])
        r = rp.unique(arr)
        n = np.unique(np.array([3, 1, 4, 1, 5, 9, 2, 6, 5]))
        assert_eq(r, n)

    def test_unique_2d(self):
        arr = rp.asarray([[1, 2], [2, 1]])
        r = rp.unique(arr)
        n = np.unique(np.array([[1, 2], [2, 1]]))
        assert_eq(r, n)
