"""Tests for newly added functions: bincount, percentile, convolve, choice."""

import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestBincount:
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
    def test_median(self):
        r = rp.quantile(rp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0.5)
        n = np.quantile([1, 2, 3, 4, 5], 0.5)
        assert abs(float(r[0]) - n) < 1e-10


class TestConvolve:
    def test_full_mode(self):
        a = rp.array([1.0, 2.0, 3.0])
        v = rp.array([0.5, 0.5])
        r = rp.convolve(a, v, 'full')
        n = np.convolve([1, 2, 3], [0.5, 0.5], 'full')
        assert_eq(r, n)

    def test_same_mode(self):
        a = rp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = rp.array([0.25, 0.5, 0.25])
        r = rp.convolve(a, v, 'same')
        n = np.convolve([1, 2, 3, 4, 5], [0.25, 0.5, 0.25], 'same')
        assert_eq(r, n)

    def test_valid_mode(self):
        a = rp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = rp.array([1.0, 2.0, 1.0])
        r = rp.convolve(a, v, 'valid')
        n = np.convolve([1, 2, 3, 4, 5], [1, 2, 1], 'valid')
        assert_eq(r, n)


class TestChoice:
    def test_from_range(self):
        rng = rp.random.default_rng(42)
        result = rng.choice(10, size=5)
        assert result.shape == (5,)
        # Check all values in range
        for val in np.array(result):
            assert 0 <= val < 10

    def test_from_list(self):
        rng = rp.random.default_rng(42)
        result = rng.choice([-1, 1], size=100)
        # Check all values are either -1 or 1
        vals = set(np.array(result).astype(int))
        assert vals <= {-1, 1}

    def test_without_replacement(self):
        rng = rp.random.default_rng(42)
        result = rng.choice(10, size=5, replace=False)
        # All values should be unique
        arr = np.array(result)
        assert len(set(arr)) == 5


class TestConcatenate:
    def test_with_lists(self):
        x = rp.array([1.0, 2.0, 3.0])
        r = rp.concatenate([[0.5], x, [4.0]])
        n = np.concatenate([[0.5], [1, 2, 3], [4.0]])
        assert_eq(r, n)

    def test_with_scalars(self):
        x = rp.array([1.0, 2.0, 3.0])
        r = rp.concatenate([[x[0]], x[1:], [x[-1]]])
        n = np.concatenate([[1], [2, 3], [3]])
        assert_eq(r, n)
