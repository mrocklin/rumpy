"""Tests for asarray - converting various inputs to arrays."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestAsarrayFromList:
    """Test creating arrays from Python lists."""

    def test_1d_list(self):
        r = rp.asarray([1.0, 2.0, 3.0])
        n = np.asarray([1.0, 2.0, 3.0])
        assert_eq(r, n)

    def test_1d_list_int(self):
        r = rp.asarray([1, 2, 3])
        n = np.asarray([1, 2, 3], dtype=np.float64)
        assert_eq(r, n)

    def test_2d_list(self):
        r = rp.asarray([[1, 2, 3], [4, 5, 6]])
        n = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        assert_eq(r, n)

    def test_3d_list(self):
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        r = rp.asarray(data)
        n = np.asarray(data, dtype=np.float64)
        assert_eq(r, n)

    def test_empty_list(self):
        r = rp.asarray([])
        n = np.asarray([], dtype=np.float64)
        assert_eq(r, n)

    def test_dtype_override(self):
        r = rp.asarray([1, 2, 3], dtype="int64")
        n = np.asarray([1, 2, 3], dtype=np.int64)
        assert_eq(r, n)


class TestAsarrayFromNumpy:
    """Test creating arrays from numpy arrays via __array_interface__."""

    def test_1d_numpy(self):
        n = np.arange(10, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_2d_numpy(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_3d_numpy(self):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_numpy_float32(self):
        n = np.arange(10, dtype=np.float32)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_numpy_int64(self):
        n = np.arange(10, dtype=np.int64)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_numpy_int32(self):
        n = np.arange(10, dtype=np.int32)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_numpy_transposed(self):
        """Ensure we correctly handle non-C-contiguous arrays."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4).T
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_numpy_sliced(self):
        """Ensure we correctly handle strided views."""
        n = np.arange(20, dtype=np.float64)[::2]
        r = rp.asarray(n)
        assert_eq(r, n)


class TestAsarrayFromRumpy:
    """Test asarray with rumpy arrays."""

    def test_rumpy_passthrough(self):
        orig = rp.arange(10)
        r = rp.asarray(orig)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r, n)

    def test_rumpy_2d(self):
        orig = rp.arange(12).reshape([3, 4])
        r = rp.asarray(orig)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r, n)


class TestAsarrayFromIterable:
    """Test asarray with generic iterables."""

    def test_tuple(self):
        r = rp.asarray((1.0, 2.0, 3.0))
        n = np.asarray((1.0, 2.0, 3.0))
        assert_eq(r, n)

    def test_range(self):
        r = rp.asarray(range(5))
        n = np.asarray(range(5), dtype=np.float64)
        assert_eq(r, n)
