"""Tests for statistical functions and sorting (Phase C)."""

import numpy as np
import rumpy

from helpers import assert_eq


class TestVar:
    """Test variance."""

    def test_var_1d(self):
        arr = rumpy.arange(10)
        r = arr.var()
        n = np.arange(10).var()
        assert abs(r - n) < 1e-10

    def test_var_2d(self):
        arr = rumpy.arange(12).reshape(3, 4)
        r = arr.var()
        n = np.arange(12).reshape(3, 4).var()
        assert abs(r - n) < 1e-10

    def test_var_axis0(self):
        arr = rumpy.arange(12, dtype="float64").reshape(3, 4)
        r = arr.var(axis=0)
        n = np.arange(12, dtype=np.float64).reshape(3, 4).var(axis=0)
        assert_eq(r, n)

    def test_var_axis1(self):
        arr = rumpy.arange(12, dtype="float64").reshape(3, 4)
        r = arr.var(axis=1)
        n = np.arange(12, dtype=np.float64).reshape(3, 4).var(axis=1)
        assert_eq(r, n)


class TestStd:
    """Test standard deviation."""

    def test_std_1d(self):
        arr = rumpy.arange(10, dtype="float64")
        r = arr.std()
        n = np.arange(10, dtype=np.float64).std()
        assert abs(r - n) < 1e-10

    def test_std_2d(self):
        arr = rumpy.arange(12, dtype="float64").reshape(3, 4)
        r = arr.std()
        n = np.arange(12, dtype=np.float64).reshape(3, 4).std()
        assert abs(r - n) < 1e-10

    def test_std_axis0(self):
        arr = rumpy.arange(12, dtype="float64").reshape(3, 4)
        r = arr.std(axis=0)
        n = np.arange(12, dtype=np.float64).reshape(3, 4).std(axis=0)
        assert_eq(r, n)


class TestArgmax:
    """Test argmax."""

    def test_argmax_1d(self):
        arr = rumpy.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        r = arr.argmax()
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6]).argmax()
        assert r == n

    def test_argmax_2d(self):
        arr = rumpy.arange(12).reshape(3, 4)
        r = arr.argmax()
        n = np.arange(12).reshape(3, 4).argmax()
        assert r == n


class TestArgmin:
    """Test argmin."""

    def test_argmin_1d(self):
        arr = rumpy.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        r = arr.argmin()
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6]).argmin()
        assert r == n

    def test_argmin_2d(self):
        arr = rumpy.arange(12).reshape(3, 4)
        r = arr.argmin()
        n = np.arange(12).reshape(3, 4).argmin()
        assert r == n


class TestSort:
    """Test sort."""

    def test_sort_1d(self):
        arr = rumpy.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        r = rumpy.sort(arr)
        n = np.sort(np.array([3, 1, 4, 1, 5, 9, 2, 6]))
        assert_eq(r, n)

    def test_sort_2d_flattened(self):
        arr = rumpy.asarray([[3, 1], [4, 2]])
        r = rumpy.sort(arr, axis=None)
        n = np.sort(np.array([[3, 1], [4, 2]]), axis=None)
        assert_eq(r, n)


class TestArgsort:
    """Test argsort."""

    def test_argsort_1d(self):
        arr = rumpy.asarray([3, 1, 4, 1, 5])
        r = rumpy.argsort(arr)
        n = np.argsort(np.array([3, 1, 4, 1, 5]))
        assert_eq(r, n)


class TestMoment:
    """Test central moments."""

    def test_moment_2_equals_var(self):
        """Moment of order 2 should equal variance."""
        arr = rumpy.arange(20, dtype="float64")
        m2 = arr.moment(2)
        v = arr.var()
        assert abs(m2 - v) < 1e-10

    def test_moment_axis(self):
        """Moment along axis."""
        arr = rumpy.arange(12, dtype="float64").reshape(3, 4)
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
        arr = rumpy.arange(-5, 6, dtype="float64")
        s = arr.skew()
        assert abs(s) < 1e-10

    def test_skew_positive(self):
        """Right-skewed distribution should have positive skewness."""
        # Heavy right tail
        arr = rumpy.asarray([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 10.0])
        s = arr.skew()
        assert s > 0

    def test_skew_negative(self):
        """Left-skewed distribution should have negative skewness."""
        # Heavy left tail
        arr = rumpy.asarray([-10.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0])
        s = arr.skew()
        assert s < 0


class TestKurtosis:
    """Test kurtosis (Fisher's definition: excess kurtosis, normal = 0)."""

    def test_kurtosis_uniform(self):
        """Uniform distribution has negative excess kurtosis (-1.2)."""
        arr = rumpy.arange(1000, dtype="float64")
        k = arr.kurtosis()
        # Uniform: kurtosis = -6/5 = -1.2
        assert abs(k - (-1.2)) < 0.01

    def test_kurtosis_leptokurtic(self):
        """Distribution with heavy tails should have positive kurtosis."""
        # Outliers create heavy tails
        arr = rumpy.asarray([0.0] * 100 + [-10.0, 10.0])
        k = arr.kurtosis()
        assert k > 0


class TestUnique:
    """Test unique."""

    def test_unique_1d(self):
        arr = rumpy.asarray([3, 1, 4, 1, 5, 9, 2, 6, 5])
        r = rumpy.unique(arr)
        n = np.unique(np.array([3, 1, 4, 1, 5, 9, 2, 6, 5]))
        assert_eq(r, n)

    def test_unique_2d(self):
        arr = rumpy.asarray([[1, 2], [2, 1]])
        r = rumpy.unique(arr)
        n = np.unique(np.array([[1, 2], [2, 1]]))
        assert_eq(r, n)
