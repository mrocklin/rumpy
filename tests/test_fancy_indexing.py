"""Tests for fancy (integer array) indexing."""

import numpy as np
import rumpy

from helpers import assert_eq


class TestFancyIndexing1D:
    """Test fancy indexing on 1D arrays."""

    def test_select_elements(self):
        arr = rumpy.arange(10)
        idx = rumpy.asarray([0, 2, 4])
        r = arr[idx]
        n = np.arange(10)[[0, 2, 4]]
        assert_eq(r, n)

    def test_select_with_duplicates(self):
        arr = rumpy.arange(5)
        idx = rumpy.asarray([0, 0, 1, 1, 2])
        r = arr[idx]
        n = np.arange(5)[[0, 0, 1, 1, 2]]
        assert_eq(r, n)

    def test_select_reversed(self):
        arr = rumpy.arange(5)
        idx = rumpy.asarray([4, 3, 2, 1, 0])
        r = arr[idx]
        n = np.arange(5)[[4, 3, 2, 1, 0]]
        assert_eq(r, n)

    def test_select_single(self):
        arr = rumpy.arange(10)
        idx = rumpy.asarray([5])
        r = arr[idx]
        n = np.arange(10)[[5]]
        assert_eq(r, n)

    def test_negative_indices(self):
        arr = rumpy.arange(10)
        idx = rumpy.asarray([-1, -2, -3])
        r = arr[idx]
        n = np.arange(10)[[-1, -2, -3]]
        assert_eq(r, n)


class TestFancyIndexing2D:
    """Test fancy indexing on 2D arrays (row selection)."""

    def test_select_rows(self):
        arr = rumpy.arange(12).reshape([3, 4])
        idx = rumpy.asarray([0, 2])
        r = arr[idx]
        n = np.arange(12).reshape(3, 4)[[0, 2]]
        assert_eq(r, n)

    def test_select_rows_reversed(self):
        arr = rumpy.arange(12).reshape([3, 4])
        idx = rumpy.asarray([2, 1, 0])
        r = arr[idx]
        n = np.arange(12).reshape(3, 4)[[2, 1, 0]]
        assert_eq(r, n)

    def test_select_rows_with_duplicates(self):
        arr = rumpy.arange(12).reshape([3, 4])
        idx = rumpy.asarray([0, 0, 2, 2])
        r = arr[idx]
        n = np.arange(12).reshape(3, 4)[[0, 0, 2, 2]]
        assert_eq(r, n)

    def test_select_single_row(self):
        arr = rumpy.arange(12).reshape([3, 4])
        idx = rumpy.asarray([1])
        r = arr[idx]
        n = np.arange(12).reshape(3, 4)[[1]]
        assert_eq(r, n)


class TestFancyIndexingDtype:
    """Test that fancy indexing preserves dtype."""

    def test_preserves_dtype_int32(self):
        arr = rumpy.arange(10, dtype="int32")
        idx = rumpy.asarray([0, 2, 4])
        r = arr[idx]
        assert r.dtype == "int32"

    def test_preserves_dtype_float32(self):
        arr = rumpy.arange(10, dtype="float32")
        idx = rumpy.asarray([0, 2, 4])
        r = arr[idx]
        assert r.dtype == "float32"


class TestFancyIndexingEmpty:
    """Test edge cases with empty selections."""

    def test_empty_index(self):
        arr = rumpy.arange(10)
        idx = rumpy.asarray([]).reshape([0])
        r = arr[idx]
        assert r.size == 0
        assert list(r.shape) == [0]

    def test_empty_2d_index(self):
        arr = rumpy.arange(12).reshape([3, 4])
        idx = rumpy.asarray([]).reshape([0])
        r = arr[idx]
        assert list(r.shape) == [0, 4]
