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
