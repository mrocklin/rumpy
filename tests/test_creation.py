"""Tests for array creation functions."""

import numpy as np
import pytest

import rumpy
from helpers import assert_eq


class TestZeros:
    """Test rumpy.zeros against np.zeros."""

    def test_1d(self):
        r = rumpy.zeros([10])
        n = np.zeros(10)
        assert_eq(r, n)

    def test_2d(self):
        r = rumpy.zeros([3, 4])
        n = np.zeros((3, 4))
        assert_eq(r, n)

    def test_3d(self):
        r = rumpy.zeros([2, 3, 4])
        n = np.zeros((2, 3, 4))
        assert_eq(r, n)

    def test_dtype_float32(self):
        r = rumpy.zeros([10], dtype="float32")
        n = np.zeros(10, dtype=np.float32)
        assert_eq(r, n)

    def test_dtype_float64(self):
        r = rumpy.zeros([10], dtype="float64")
        n = np.zeros(10, dtype=np.float64)
        assert_eq(r, n)

    def test_dtype_int32(self):
        r = rumpy.zeros([10], dtype="int32")
        n = np.zeros(10, dtype=np.int32)
        assert_eq(r, n)

    def test_dtype_int64(self):
        r = rumpy.zeros([10], dtype="int64")
        n = np.zeros(10, dtype=np.int64)
        assert_eq(r, n)

    def test_dtype_bool(self):
        r = rumpy.zeros([10], dtype="bool")
        n = np.zeros(10, dtype=bool)
        assert_eq(r, n)


class TestOnes:
    """Test rumpy.ones against np.ones."""

    def test_1d(self):
        r = rumpy.ones([10])
        n = np.ones(10)
        assert_eq(r, n)

    def test_2d(self):
        r = rumpy.ones([3, 4])
        n = np.ones((3, 4))
        assert_eq(r, n)

    def test_dtype_float32(self):
        r = rumpy.ones([10], dtype="float32")
        n = np.ones(10, dtype=np.float32)
        assert_eq(r, n)

    def test_dtype_float64(self):
        r = rumpy.ones([10], dtype="float64")
        n = np.ones(10, dtype=np.float64)
        assert_eq(r, n)

    def test_dtype_int32(self):
        r = rumpy.ones([10], dtype="int32")
        n = np.ones(10, dtype=np.int32)
        assert_eq(r, n)

    def test_dtype_int64(self):
        r = rumpy.ones([10], dtype="int64")
        n = np.ones(10, dtype=np.int64)
        assert_eq(r, n)

    def test_dtype_bool(self):
        r = rumpy.ones([10], dtype="bool")
        n = np.ones(10, dtype=bool)
        assert_eq(r, n)


class TestArrayProperties:
    """Test array properties match numpy."""

    def test_shape(self):
        r = rumpy.zeros([3, 4, 5])
        assert r.shape == [3, 4, 5]

    def test_ndim(self):
        r = rumpy.zeros([3, 4, 5])
        assert r.ndim == 3

    def test_size(self):
        r = rumpy.zeros([3, 4, 5])
        assert r.size == 60

    def test_itemsize_float64(self):
        r = rumpy.zeros([10], dtype="float64")
        assert r.itemsize == 8

    def test_itemsize_float32(self):
        r = rumpy.zeros([10], dtype="float32")
        assert r.itemsize == 4

    def test_nbytes(self):
        r = rumpy.zeros([10], dtype="float64")
        assert r.nbytes == 80

    def test_strides_c_order(self):
        r = rumpy.zeros([3, 4])
        # C-order: last dimension has stride = itemsize
        assert r.strides == [32, 8]  # 4*8, 1*8 for float64
