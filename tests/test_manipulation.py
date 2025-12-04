"""Tests for array manipulation functions (Phase B)."""

import numpy as np
import rumpy

from helpers import assert_eq


class TestCopy:
    """Test copy() method."""

    def test_copy_basic(self):
        arr = rumpy.arange(10)
        c = arr.copy()
        assert_eq(c, np.arange(10))

    def test_copy_2d(self):
        arr = rumpy.arange(12).reshape([3, 4])
        c = arr.copy()
        assert_eq(c, np.arange(12).reshape(3, 4))

    def test_copy_preserves_dtype(self):
        arr = rumpy.arange(10, dtype="int32")
        c = arr.copy()
        assert c.dtype == "int32"


class TestAstype:
    """Test astype() method."""

    def test_float64_to_int32(self):
        arr = rumpy.arange(5)
        c = arr.astype("int32")
        assert c.dtype == "int32"
        assert_eq(c, np.arange(5).astype("int32"))

    def test_int32_to_float64(self):
        arr = rumpy.arange(5, dtype="int32")
        c = arr.astype("float64")
        assert c.dtype == "float64"
        assert_eq(c, np.arange(5, dtype="int32").astype("float64"))


class TestSqueeze:
    """Test squeeze() function and method."""

    def test_squeeze_basic(self):
        arr = rumpy.arange(5).reshape([1, 5, 1])
        r = arr.squeeze()
        n = np.arange(5).reshape(1, 5, 1).squeeze()
        assert_eq(r, n)

    def test_squeeze_module(self):
        arr = rumpy.arange(5).reshape([1, 5])
        r = rumpy.squeeze(arr)
        n = np.squeeze(np.arange(5).reshape(1, 5))
        assert_eq(r, n)

    def test_squeeze_noop(self):
        arr = rumpy.arange(6).reshape([2, 3])
        r = arr.squeeze()
        n = np.arange(6).reshape(2, 3).squeeze()
        assert_eq(r, n)


class TestExpandDims:
    """Test expand_dims() function."""

    def test_expand_dims_axis0(self):
        arr = rumpy.arange(5)
        r = rumpy.expand_dims(arr, 0)
        n = np.expand_dims(np.arange(5), 0)
        assert_eq(r, n)

    def test_expand_dims_axis1(self):
        arr = rumpy.arange(5)
        r = rumpy.expand_dims(arr, 1)
        n = np.expand_dims(np.arange(5), 1)
        assert_eq(r, n)

    def test_expand_dims_negative(self):
        arr = rumpy.arange(5)
        r = rumpy.expand_dims(arr, -1)
        n = np.expand_dims(np.arange(5), -1)
        assert_eq(r, n)


class TestConcatenate:
    """Test concatenate() function."""

    def test_concatenate_1d(self):
        a = rumpy.arange(3)
        b = rumpy.arange(3, 6)
        r = rumpy.concatenate([a, b])
        n = np.concatenate([np.arange(3), np.arange(3, 6)])
        assert_eq(r, n)

    def test_concatenate_2d_axis0(self):
        a = rumpy.arange(6).reshape([2, 3])
        b = rumpy.arange(6, 12).reshape([2, 3])
        r = rumpy.concatenate([a, b], axis=0)
        n = np.concatenate([np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)], axis=0)
        assert_eq(r, n)

    def test_concatenate_2d_axis1(self):
        a = rumpy.arange(6).reshape([2, 3])
        b = rumpy.arange(6, 10).reshape([2, 2])
        r = rumpy.concatenate([a, b], axis=1)
        n = np.concatenate([np.arange(6).reshape(2, 3), np.arange(6, 10).reshape(2, 2)], axis=1)
        assert_eq(r, n)


class TestStack:
    """Test stack() function."""

    def test_stack_1d_axis0(self):
        a = rumpy.arange(3)
        b = rumpy.arange(3, 6)
        r = rumpy.stack([a, b], axis=0)
        n = np.stack([np.arange(3), np.arange(3, 6)], axis=0)
        assert_eq(r, n)

    def test_stack_1d_axis1(self):
        a = rumpy.arange(3)
        b = rumpy.arange(3, 6)
        r = rumpy.stack([a, b], axis=1)
        n = np.stack([np.arange(3), np.arange(3, 6)], axis=1)
        assert_eq(r, n)


class TestVstack:
    """Test vstack() function."""

    def test_vstack_1d(self):
        a = rumpy.arange(3)
        b = rumpy.arange(3)
        r = rumpy.vstack([a, b])
        n = np.vstack([np.arange(3), np.arange(3)])
        assert_eq(r, n)

    def test_vstack_2d(self):
        a = rumpy.arange(6).reshape([2, 3])
        b = rumpy.arange(6).reshape([2, 3])
        r = rumpy.vstack([a, b])
        n = np.vstack([np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3)])
        assert_eq(r, n)


class TestHstack:
    """Test hstack() function."""

    def test_hstack_1d(self):
        a = rumpy.arange(3)
        b = rumpy.arange(3)
        r = rumpy.hstack([a, b])
        n = np.hstack([np.arange(3), np.arange(3)])
        assert_eq(r, n)

    def test_hstack_2d(self):
        a = rumpy.arange(6).reshape([2, 3])
        b = rumpy.arange(6).reshape([2, 3])
        r = rumpy.hstack([a, b])
        n = np.hstack([np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3)])
        assert_eq(r, n)


class TestSplit:
    """Test split() function."""

    def test_split_1d(self):
        arr = rumpy.arange(9)
        parts = rumpy.split(arr, 3)
        n_parts = np.split(np.arange(9), 3)
        assert len(parts) == 3
        for r, n in zip(parts, n_parts):
            assert_eq(r, n)

    def test_split_2d_axis0(self):
        arr = rumpy.arange(12).reshape([4, 3])
        parts = rumpy.split(arr, 2, axis=0)
        n_parts = np.split(np.arange(12).reshape(4, 3), 2, axis=0)
        assert len(parts) == 2
        for r, n in zip(parts, n_parts):
            assert_eq(r, n)


class TestArraySplit:
    """Test array_split() function."""

    def test_array_split_even(self):
        arr = rumpy.arange(9)
        parts = rumpy.array_split(arr, 3)
        n_parts = np.array_split(np.arange(9), 3)
        assert len(parts) == 3
        for r, n in zip(parts, n_parts):
            assert_eq(r, n)

    def test_array_split_uneven(self):
        arr = rumpy.arange(10)
        parts = rumpy.array_split(arr, 3)
        n_parts = np.array_split(np.arange(10), 3)
        assert len(parts) == 3
        for r, n in zip(parts, n_parts):
            assert_eq(r, n)
