"""Tests for window functions (Stream 28)."""

import numpy as np
import pytest
import rumpy as rp

from helpers import assert_eq


WINDOW_FUNCTIONS = ["bartlett", "blackman", "hamming", "hanning"]


class TestWindowFunctions:
    """Test basic window functions."""

    @pytest.mark.parametrize("func_name", WINDOW_FUNCTIONS)
    @pytest.mark.parametrize("M", [0, 1, 2, 5, 10, 51, 100])
    def test_sizes(self, func_name, M):
        """Test various window sizes."""
        np_func = getattr(np, func_name)
        rp_func = getattr(rp, func_name)
        assert_eq(rp_func(M), np_func(M))

    @pytest.mark.parametrize("func_name", WINDOW_FUNCTIONS)
    def test_dtype_is_float64(self, func_name):
        """Window functions always return float64."""
        rp_func = getattr(rp, func_name)
        result = rp_func(10)
        assert result.dtype == np.dtype("float64")

    @pytest.mark.parametrize("func_name", WINDOW_FUNCTIONS)
    def test_symmetry(self, func_name):
        """Window functions are symmetric."""
        rp_func = getattr(rp, func_name)
        result = rp_func(11)
        n = np.asarray(result)
        # Check symmetric
        assert_eq(n, n[::-1])

    @pytest.mark.parametrize("func_name", WINDOW_FUNCTIONS)
    def test_negative_size(self, func_name):
        """Negative size returns empty array."""
        np_func = getattr(np, func_name)
        rp_func = getattr(rp, func_name)
        assert_eq(rp_func(-1), np_func(-1))


class TestKaiser:
    """Test kaiser window function."""

    @pytest.mark.parametrize("M", [0, 1, 2, 5, 10, 51, 100])
    @pytest.mark.parametrize("beta", [0.0, 1.0, 5.0, 14.0])
    def test_sizes_and_beta(self, M, beta):
        """Test various window sizes and beta values."""
        assert_eq(rp.kaiser(M, beta), np.kaiser(M, beta))

    def test_dtype_is_float64(self):
        """Kaiser always returns float64."""
        result = rp.kaiser(10, 5.0)
        assert result.dtype == np.dtype("float64")

    def test_symmetry(self):
        """Kaiser window is symmetric."""
        result = rp.kaiser(11, 5.0)
        n = np.asarray(result)
        assert_eq(n, n[::-1])

    def test_negative_size(self):
        """Negative size returns empty array."""
        assert_eq(rp.kaiser(-1, 5.0), np.kaiser(-1, 5.0))

    def test_beta_zero_is_rectangular(self):
        """Beta=0 gives a rectangular window (all ones)."""
        result = rp.kaiser(10, 0.0)
        assert_eq(result, np.ones(10))
