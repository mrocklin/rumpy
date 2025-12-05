"""Tests for array creation functions."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestZeros:
    """Test rp.zeros against np.zeros."""

    def test_1d(self):
        r = rp.zeros(10)
        n = np.zeros(10)
        assert_eq(r, n)

    def test_2d(self):
        r = rp.zeros((3, 4))
        n = np.zeros((3, 4))
        assert_eq(r, n)

    def test_3d(self):
        r = rp.zeros((2, 3, 4))
        n = np.zeros((2, 3, 4))
        assert_eq(r, n)

    def test_dtype_float32(self):
        r = rp.zeros(10, dtype="float32")
        n = np.zeros(10, dtype=np.float32)
        assert_eq(r, n)

    def test_dtype_float64(self):
        r = rp.zeros(10, dtype="float64")
        n = np.zeros(10, dtype=np.float64)
        assert_eq(r, n)

    def test_dtype_int32(self):
        r = rp.zeros(10, dtype="int32")
        n = np.zeros(10, dtype=np.int32)
        assert_eq(r, n)

    def test_dtype_int64(self):
        r = rp.zeros(10, dtype="int64")
        n = np.zeros(10, dtype=np.int64)
        assert_eq(r, n)

    def test_dtype_bool(self):
        r = rp.zeros(10, dtype="bool")
        n = np.zeros(10, dtype=bool)
        assert_eq(r, n)


class TestOnes:
    """Test rp.ones against np.ones."""

    def test_1d(self):
        r = rp.ones(10)
        n = np.ones(10)
        assert_eq(r, n)

    def test_2d(self):
        r = rp.ones((3, 4))
        n = np.ones((3, 4))
        assert_eq(r, n)

    def test_dtype_float32(self):
        r = rp.ones(10, dtype="float32")
        n = np.ones(10, dtype=np.float32)
        assert_eq(r, n)

    def test_dtype_float64(self):
        r = rp.ones(10, dtype="float64")
        n = np.ones(10, dtype=np.float64)
        assert_eq(r, n)

    def test_dtype_int32(self):
        r = rp.ones(10, dtype="int32")
        n = np.ones(10, dtype=np.int32)
        assert_eq(r, n)

    def test_dtype_int64(self):
        r = rp.ones(10, dtype="int64")
        n = np.ones(10, dtype=np.int64)
        assert_eq(r, n)

    def test_dtype_bool(self):
        r = rp.ones(10, dtype="bool")
        n = np.ones(10, dtype=bool)
        assert_eq(r, n)


class TestArrayProperties:
    """Test array properties match numpy."""

    def test_shape(self):
        r = rp.zeros((3, 4, 5))
        assert r.shape == (3, 4, 5)

    def test_ndim(self):
        r = rp.zeros((3, 4, 5))
        assert r.ndim == 3

    def test_size(self):
        r = rp.zeros((3, 4, 5))
        assert r.size == 60

    def test_itemsize_float64(self):
        r = rp.zeros(10, dtype="float64")
        assert r.itemsize == 8

    def test_itemsize_float32(self):
        r = rp.zeros(10, dtype="float32")
        assert r.itemsize == 4

    def test_nbytes(self):
        r = rp.zeros(10, dtype="float64")
        assert r.nbytes == 80

    def test_strides_c_order(self):
        r = rp.zeros((3, 4))
        # C-order: last dimension has stride = itemsize
        assert r.strides == (32, 8)  # 4*8, 1*8 for float64


class TestEmpty:
    """Test rp.empty against np.empty."""

    def test_1d(self):
        r = rp.empty(10)
        n = np.empty(10)
        assert r.shape == n.shape
        assert r.dtype == "float64"

    def test_2d(self):
        r = rp.empty((3, 4))
        n = np.empty((3, 4))
        assert r.shape == n.shape

    def test_dtype(self):
        r = rp.empty(10, dtype="float32")
        assert r.dtype == "float32"


class TestZerosLike:
    """Test rp.zeros_like against np.zeros_like."""

    def test_basic(self):
        arr = rp.ones((3, 4))
        r = rp.zeros_like(arr)
        n = np.zeros_like(np.ones((3, 4)))
        assert_eq(r, n)

    def test_preserves_dtype(self):
        arr = rp.ones((3, 4), dtype="float32")
        r = rp.zeros_like(arr)
        assert r.dtype == "float32"
        assert r.shape == (3, 4)

    def test_override_dtype(self):
        arr = rp.ones((3, 4), dtype="float32")
        r = rp.zeros_like(arr, dtype="float64")
        assert r.dtype == "float64"


class TestOnesLike:
    """Test rp.ones_like against np.ones_like."""

    def test_basic(self):
        arr = rp.zeros((3, 4))
        r = rp.ones_like(arr)
        n = np.ones_like(np.zeros((3, 4)))
        assert_eq(r, n)

    def test_preserves_dtype(self):
        arr = rp.zeros((3, 4), dtype="int32")
        r = rp.ones_like(arr)
        assert r.dtype == "int32"
        assert r.shape == (3, 4)

    def test_override_dtype(self):
        arr = rp.zeros((3, 4), dtype="int32")
        r = rp.ones_like(arr, dtype="float64")
        assert r.dtype == "float64"


class TestEmptyLike:
    """Test rp.empty_like against np.empty_like."""

    def test_basic(self):
        arr = rp.ones((3, 4))
        r = rp.empty_like(arr)
        assert r.shape == (3, 4)
        assert r.dtype == "float64"

    def test_preserves_dtype(self):
        arr = rp.ones((3, 4), dtype="float32")
        r = rp.empty_like(arr)
        assert r.dtype == "float32"

    def test_override_dtype(self):
        arr = rp.ones((3, 4), dtype="float32")
        r = rp.empty_like(arr, dtype="int64")
        assert r.dtype == "int64"


class TestCopyModule:
    """Test rp.copy module function."""

    def test_basic(self):
        arr = rp.arange(10)
        r = rp.copy(arr)
        n = np.copy(np.arange(10, dtype=np.float64))
        assert_eq(r, n)

    def test_2d(self):
        arr = rp.arange(12).reshape(3, 4)
        r = rp.copy(arr)
        n = np.copy(np.arange(12, dtype=np.float64).reshape(3, 4))
        assert_eq(r, n)

    def test_preserves_dtype(self):
        arr = rp.arange(10, dtype="int32")
        r = rp.copy(arr)
        assert r.dtype == "int32"
