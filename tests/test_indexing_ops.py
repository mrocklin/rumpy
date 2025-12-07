"""Tests for indexing operations against numpy.

Covers:
- take: take elements along axis
- take_along_axis: take using index array along axis
- compress: select elements using boolean mask along axis
- searchsorted: find insertion points for sorted array
- argwhere: indices where condition is true
- flatnonzero: indices of non-zero elements in flattened array
- put: replace elements at indices
- put_along_axis: put values using index array
- choose: construct array from index array and choices
"""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestTake:
    """Test np.take - take elements along axis."""

    def test_take_1d(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        indices = [0, 2, 4, 6, 8]
        assert_eq(rp.take(r, indices), np.take(n, indices))

    def test_take_1d_negative_indices(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        indices = [-1, -2, -3]
        assert_eq(rp.take(r, indices), np.take(n, indices))

    def test_take_1d_with_duplicates(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        indices = [0, 0, 1, 1, 2]
        assert_eq(rp.take(r, indices), np.take(n, indices))

    def test_take_2d_axis0(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        indices = [0, 2]
        assert_eq(rp.take(r, indices, axis=0), np.take(n, indices, axis=0))

    def test_take_2d_axis1(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        indices = [0, 2, 3]
        assert_eq(rp.take(r, indices, axis=1), np.take(n, indices, axis=1))

    def test_take_3d_axis0(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        indices = [1, 0, 1]
        assert_eq(rp.take(r, indices, axis=0), np.take(n, indices, axis=0))

    def test_take_3d_axis1(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        indices = [0, 2]
        assert_eq(rp.take(r, indices, axis=1), np.take(n, indices, axis=1))

    def test_take_empty_indices(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        indices = []
        assert_eq(rp.take(r, indices), np.take(n, indices))


class TestTakeAlongAxis:
    """Test np.take_along_axis - take using index array along axis."""

    def test_take_along_axis_1d(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        indices = rp.asarray([0, 2, 4])
        n_indices = np.array([0, 2, 4])
        assert_eq(
            rp.take_along_axis(r, indices, axis=0),
            np.take_along_axis(n, n_indices, axis=0)
        )

    def test_take_along_axis_2d_axis0(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        # Indices shape must match array except along take axis
        indices = rp.asarray([[0, 1, 0, 2], [2, 0, 1, 0]])  # shape (2, 4)
        n_indices = np.array([[0, 1, 0, 2], [2, 0, 1, 0]])
        assert_eq(
            rp.take_along_axis(r, indices, axis=0),
            np.take_along_axis(n, n_indices, axis=0)
        )

    def test_take_along_axis_2d_axis1(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        # Indices shape must match array except along take axis
        indices = rp.asarray([[0, 2], [1, 3], [0, 1]])  # shape (3, 2)
        n_indices = np.array([[0, 2], [1, 3], [0, 1]])
        assert_eq(
            rp.take_along_axis(r, indices, axis=1),
            np.take_along_axis(n, n_indices, axis=1)
        )

    def test_take_along_axis_sorted_indices(self):
        """Common use case: using argsort result."""
        r = rp.asarray([3.0, 1.0, 4.0, 1.0, 5.0])
        n = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        r_sorted_idx = rp.argsort(r)
        n_sorted_idx = np.argsort(n)
        assert_eq(
            rp.take_along_axis(r, r_sorted_idx, axis=0),
            np.take_along_axis(n, n_sorted_idx, axis=0)
        )


class TestCompress:
    """Test np.compress - select elements using boolean mask along axis."""

    def test_compress_1d(self):
        r = rp.arange(6)
        n = np.arange(6, dtype=np.float64)
        condition = [True, False, True, False, True, False]
        assert_eq(rp.compress(condition, r), np.compress(condition, n))

    def test_compress_2d_axis0(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        condition = [True, False, True]
        assert_eq(
            rp.compress(condition, r, axis=0),
            np.compress(condition, n, axis=0)
        )

    def test_compress_2d_axis1(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        condition = [True, True, False, True]
        assert_eq(
            rp.compress(condition, r, axis=1),
            np.compress(condition, n, axis=1)
        )

    def test_compress_all_false(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        condition = [False, False, False, False, False]
        assert_eq(rp.compress(condition, r), np.compress(condition, n))

    def test_compress_all_true(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        condition = [True, True, True, True, True]
        assert_eq(rp.compress(condition, r), np.compress(condition, n))


class TestSearchsorted:
    """Test np.searchsorted - find insertion points for sorted array."""

    def test_searchsorted_1d(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = rp.asarray([0.5, 1.5, 2.5, 5.5])
        n_v = np.array([0.5, 1.5, 2.5, 5.5])
        assert_eq(rp.searchsorted(r, v), np.searchsorted(n, n_v))

    def test_searchsorted_scalar(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert int(np.asarray(rp.searchsorted(r, 2.5))) == int(np.searchsorted(n, 2.5))

    def test_searchsorted_side_left(self):
        r = rp.asarray([1.0, 2.0, 2.0, 3.0])
        n = np.array([1.0, 2.0, 2.0, 3.0])
        v = rp.asarray([2.0])
        n_v = np.array([2.0])
        assert_eq(
            rp.searchsorted(r, v, side="left"),
            np.searchsorted(n, n_v, side="left")
        )

    def test_searchsorted_side_right(self):
        r = rp.asarray([1.0, 2.0, 2.0, 3.0])
        n = np.array([1.0, 2.0, 2.0, 3.0])
        v = rp.asarray([2.0])
        n_v = np.array([2.0])
        assert_eq(
            rp.searchsorted(r, v, side="right"),
            np.searchsorted(n, n_v, side="right")
        )

    def test_searchsorted_empty(self):
        r = rp.asarray([])
        n = np.array([], dtype=np.float64)
        v = rp.asarray([1.0])
        n_v = np.array([1.0])
        assert_eq(rp.searchsorted(r, v), np.searchsorted(n, n_v))


class TestArgwhere:
    """Test np.argwhere - indices where condition is true."""

    def test_argwhere_1d(self):
        r = rp.asarray([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
        n = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
        assert_eq(rp.argwhere(r), np.argwhere(n))

    def test_argwhere_2d(self):
        r = rp.asarray([[0.0, 1.0], [2.0, 0.0]])
        n = np.array([[0.0, 1.0], [2.0, 0.0]])
        assert_eq(rp.argwhere(r), np.argwhere(n))

    def test_argwhere_all_zero(self):
        r = rp.zeros([3, 3])
        n = np.zeros((3, 3))
        assert_eq(rp.argwhere(r), np.argwhere(n))

    def test_argwhere_all_nonzero(self):
        r = rp.ones([2, 2])
        n = np.ones((2, 2))
        assert_eq(rp.argwhere(r), np.argwhere(n))

    def test_argwhere_boolean(self):
        r = rp.arange(5) > 2
        n = np.arange(5, dtype=np.float64) > 2
        assert_eq(rp.argwhere(r), np.argwhere(n))


class TestFlatnonzero:
    """Test np.flatnonzero - indices of non-zero elements in flattened array."""

    def test_flatnonzero_1d(self):
        r = rp.asarray([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
        n = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
        assert_eq(rp.flatnonzero(r), np.flatnonzero(n))

    def test_flatnonzero_2d(self):
        r = rp.asarray([[0.0, 1.0], [2.0, 0.0]])
        n = np.array([[0.0, 1.0], [2.0, 0.0]])
        assert_eq(rp.flatnonzero(r), np.flatnonzero(n))

    def test_flatnonzero_all_zero(self):
        r = rp.zeros([5])
        n = np.zeros(5)
        assert_eq(rp.flatnonzero(r), np.flatnonzero(n))

    def test_flatnonzero_all_nonzero(self):
        r = rp.arange(1, 6)  # [1, 2, 3, 4, 5]
        n = np.arange(1, 6, dtype=np.float64)
        assert_eq(rp.flatnonzero(r), np.flatnonzero(n))


class TestPut:
    """Test np.put - replace elements at indices."""

    def test_put_1d(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        rp.put(r, [0, 2], [10.0, 20.0])
        np.put(n, [0, 2], [10.0, 20.0])
        assert_eq(r, n)

    def test_put_negative_indices(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        rp.put(r, [-1, -2], [100.0, 200.0])
        np.put(n, [-1, -2], [100.0, 200.0])
        assert_eq(r, n)

    def test_put_2d_flat(self):
        """put works on flattened array."""
        r = rp.arange(6).reshape([2, 3])
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        rp.put(r, [0, 4], [10.0, 40.0])
        np.put(n, [0, 4], [10.0, 40.0])
        assert_eq(r, n)

    def test_put_scalar_value(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        rp.put(r, [0, 1, 2], 99.0)
        np.put(n, [0, 1, 2], 99.0)
        assert_eq(r, n)


class TestPutAlongAxis:
    """Test np.put_along_axis - put values using index array."""

    def test_put_along_axis_1d(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        indices = rp.asarray([0, 2, 4])
        n_indices = np.array([0, 2, 4])
        values = rp.asarray([10.0, 20.0, 30.0])
        n_values = np.array([10.0, 20.0, 30.0])
        rp.put_along_axis(r, indices, values, axis=0)
        np.put_along_axis(n, n_indices, n_values, axis=0)
        assert_eq(r, n)

    def test_put_along_axis_2d_axis0(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        indices = rp.asarray([[0, 1, 0, 2]])  # shape (1, 4)
        n_indices = np.array([[0, 1, 0, 2]])
        values = rp.asarray([[100.0, 200.0, 300.0, 400.0]])
        n_values = np.array([[100.0, 200.0, 300.0, 400.0]])
        rp.put_along_axis(r, indices, values, axis=0)
        np.put_along_axis(n, n_indices, n_values, axis=0)
        assert_eq(r, n)


class TestChoose:
    """Test np.choose - construct array from index array and choices."""

    def test_choose_basic(self):
        choices = [
            rp.asarray([0.0, 1.0, 2.0]),
            rp.asarray([10.0, 11.0, 12.0]),
            rp.asarray([20.0, 21.0, 22.0])
        ]
        n_choices = [
            np.array([0.0, 1.0, 2.0]),
            np.array([10.0, 11.0, 12.0]),
            np.array([20.0, 21.0, 22.0])
        ]
        indices = rp.asarray([0, 1, 2])
        n_indices = np.array([0, 1, 2])
        assert_eq(rp.choose(indices, choices), np.choose(n_indices, n_choices))

    def test_choose_2d_indices(self):
        choices = [
            rp.asarray([[0.0, 1.0], [2.0, 3.0]]),
            rp.asarray([[10.0, 11.0], [12.0, 13.0]])
        ]
        n_choices = [
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            np.array([[10.0, 11.0], [12.0, 13.0]])
        ]
        indices = rp.asarray([[0, 1], [1, 0]])
        n_indices = np.array([[0, 1], [1, 0]])
        assert_eq(rp.choose(indices, choices), np.choose(n_indices, n_choices))
