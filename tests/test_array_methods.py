"""Tests for ndarray methods (Stream 18)."""
import numpy as np
import pytest
import rumpy as rp

from helpers import assert_eq


class TestNonzero:
    """Tests for nonzero() method."""

    def test_nonzero_1d(self):
        arr = rp.asarray([0, 1, 0, 2, 3, 0])
        result = arr.nonzero()
        expected = np.asarray([0, 1, 0, 2, 3, 0]).nonzero()
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_nonzero_2d(self):
        arr = rp.asarray([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        result = arr.nonzero()
        expected = np.asarray([[1, 0, 0], [0, 2, 0], [0, 0, 3]]).nonzero()
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_nonzero_all_zeros(self):
        arr = rp.zeros(5)
        result = arr.nonzero()
        expected = np.zeros(5).nonzero()
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_eq(r, e)


class TestArgsort:
    """Tests for argsort() method."""

    def test_argsort_1d(self):
        arr = rp.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        n = np.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        assert_eq(arr.argsort(), n.argsort())

    def test_argsort_2d_axis0(self):
        arr = rp.asarray([[3, 1], [2, 4], [1, 3]])
        n = np.asarray([[3, 1], [2, 4], [1, 3]])
        assert_eq(arr.argsort(axis=0), n.argsort(axis=0))

    def test_argsort_2d_axis1(self):
        arr = rp.asarray([[3, 1], [2, 4], [1, 3]])
        n = np.asarray([[3, 1], [2, 4], [1, 3]])
        assert_eq(arr.argsort(axis=1), n.argsort(axis=1))

    def test_argsort_2d_none(self):
        arr = rp.asarray([[3, 1], [2, 4]])
        n = np.asarray([[3, 1], [2, 4]])
        assert_eq(arr.argsort(axis=None), n.argsort(axis=None))


class TestSort:
    """Tests for sort() method (in-place)."""

    def test_sort_1d(self):
        arr = rp.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        n = np.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        arr.sort()
        n.sort()
        assert_eq(arr, n)

    def test_sort_2d_default(self):
        arr = rp.asarray([[3, 1], [2, 4], [1, 3]])
        n = np.asarray([[3, 1], [2, 4], [1, 3]])
        arr.sort()  # sorts along last axis by default
        n.sort()
        assert_eq(arr, n)

    def test_sort_2d_axis0(self):
        arr = rp.asarray([[3, 1], [2, 4], [1, 3]])
        n = np.asarray([[3, 1], [2, 4], [1, 3]])
        arr.sort(axis=0)
        n.sort(axis=0)
        assert_eq(arr, n)


class TestSearchsorted:
    """Tests for searchsorted() method."""

    def test_searchsorted_basic(self):
        arr = rp.asarray([1, 2, 3, 4, 5])
        v = rp.asarray([2.5, 0.5, 4.0])
        n = np.asarray([1, 2, 3, 4, 5])
        nv = np.asarray([2.5, 0.5, 4.0])
        assert_eq(arr.searchsorted(v), n.searchsorted(nv))

    def test_searchsorted_right(self):
        arr = rp.asarray([1, 2, 3, 4, 5])
        v = rp.asarray([2.0, 4.0])
        n = np.asarray([1, 2, 3, 4, 5])
        nv = np.asarray([2.0, 4.0])
        assert_eq(arr.searchsorted(v, side='right'), n.searchsorted(nv, side='right'))

    def test_searchsorted_scalar(self):
        arr = rp.asarray([1, 2, 3, 4, 5])
        n = np.asarray([1, 2, 3, 4, 5])
        r = arr.searchsorted(2.5)
        e = n.searchsorted(2.5)
        assert r == e


class TestRepeat:
    """Tests for repeat() method."""

    def test_repeat_scalar(self):
        arr = rp.asarray([1, 2, 3])
        n = np.asarray([1, 2, 3])
        assert_eq(arr.repeat(2), n.repeat(2))

    def test_repeat_axis0(self):
        arr = rp.asarray([[1, 2], [3, 4]])
        n = np.asarray([[1, 2], [3, 4]])
        assert_eq(arr.repeat(2, axis=0), n.repeat(2, axis=0))

    def test_repeat_axis1(self):
        arr = rp.asarray([[1, 2], [3, 4]])
        n = np.asarray([[1, 2], [3, 4]])
        assert_eq(arr.repeat(3, axis=1), n.repeat(3, axis=1))


class TestTake:
    """Tests for take() method."""

    def test_take_1d(self):
        arr = rp.asarray([10, 20, 30, 40, 50])
        n = np.asarray([10, 20, 30, 40, 50])
        indices = rp.asarray([0, 2, 4])
        n_indices = np.asarray([0, 2, 4])
        assert_eq(arr.take(indices), n.take(n_indices))

    def test_take_2d_flat(self):
        arr = rp.asarray([[1, 2], [3, 4]])
        n = np.asarray([[1, 2], [3, 4]])
        indices = rp.asarray([0, 3])
        n_indices = np.asarray([0, 3])
        assert_eq(arr.take(indices), n.take(n_indices))

    def test_take_axis0(self):
        arr = rp.asarray([[1, 2], [3, 4], [5, 6]])
        n = np.asarray([[1, 2], [3, 4], [5, 6]])
        indices = rp.asarray([0, 2])
        n_indices = np.asarray([0, 2])
        assert_eq(arr.take(indices, axis=0), n.take(n_indices, axis=0))


class TestPut:
    """Tests for put() method."""

    def test_put_basic(self):
        arr = rp.asarray([1, 2, 3, 4, 5])
        n = np.asarray([1, 2, 3, 4, 5])
        arr.put([0, 2], [10, 30])
        n.put([0, 2], [10, 30])
        assert_eq(arr, n)

    def test_put_single(self):
        arr = rp.asarray([1, 2, 3, 4, 5])
        n = np.asarray([1, 2, 3, 4, 5])
        arr.put([1], [99])
        n.put([1], [99])
        assert_eq(arr, n)


class TestFill:
    """Tests for fill() method."""

    def test_fill_basic(self):
        arr = rp.zeros(5)
        n = np.zeros(5)
        arr.fill(7)
        n.fill(7)
        assert_eq(arr, n)

    def test_fill_2d(self):
        arr = rp.zeros((3, 4))
        n = np.zeros((3, 4))
        arr.fill(-1.5)
        n.fill(-1.5)
        assert_eq(arr, n)


class TestTobytes:
    """Tests for tobytes() method."""

    def test_tobytes_float64(self):
        arr = rp.asarray([1.0, 2.0, 3.0])
        n = np.asarray([1.0, 2.0, 3.0])
        assert arr.tobytes() == n.tobytes()

    def test_tobytes_int64(self):
        arr = rp.asarray([1, 2, 3], dtype='int64')
        n = np.asarray([1, 2, 3], dtype='int64')
        assert arr.tobytes() == n.tobytes()


class TestView:
    """Tests for view() method with dtype."""

    def test_view_float64_to_int64(self):
        arr = rp.asarray([1.0, 2.0])
        n = np.asarray([1.0, 2.0])
        r = arr.view('int64')
        e = n.view('int64')
        assert_eq(r, e)

    def test_view_int32_to_int16(self):
        arr = rp.asarray([0x00010002, 0x00030004], dtype='int32')
        n = np.asarray([0x00010002, 0x00030004], dtype='int32')
        r = arr.view('int16')
        e = n.view('int16')
        assert_eq(r, e)


class TestPartition:
    """Tests for partition() method."""

    def test_partition_basic(self):
        arr = rp.asarray([3, 4, 2, 1])
        n = np.asarray([3, 4, 2, 1])
        arr.partition(2)
        n.partition(2)
        # After partition, element at kth position should equal sorted value
        assert float(arr[2]) == float(n[2])

    def test_partition_sorted_property(self):
        arr = rp.asarray([3, 4, 2, 1, 5])
        arr.partition(2)
        # All elements before index 2 should be <= arr[2]
        # All elements after index 2 should be >= arr[2]
        pivot = float(arr[2])
        for i in range(2):
            assert float(arr[i]) <= pivot
        for i in range(3, 5):
            assert float(arr[i]) >= pivot


class TestArgpartition:
    """Tests for argpartition() method."""

    def test_argpartition_basic(self):
        arr = rp.asarray([3, 4, 2, 1, 5])
        n = np.asarray([3, 4, 2, 1, 5])
        r = arr.argpartition(2)
        e = n.argpartition(2)
        # The kth element in the partitioned array should match
        assert float(arr.take(r)[2]) == float(np.take(n, e)[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
