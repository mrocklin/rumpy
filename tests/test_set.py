"""Tests for set operations.

Parametrizes over operations and dtypes for comprehensive coverage.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import NUMERIC_DTYPES
from helpers import assert_eq, make_pair

# === Set operation categories ===

# Binary set operations (two input arrays)
BINARY_SET_OPS = ["intersect1d", "union1d", "setdiff1d", "setxor1d"]


# === Parametrized tests by category ===


class TestIsin:
    """Test isin operation (element-wise membership)."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, dtype):
        """Test all numeric dtypes."""
        a_n = np.array([1, 2, 3, 4, 5], dtype=dtype)
        test_n = np.array([2, 4], dtype=dtype)
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.isin(a_r, test_r), np.isin(a_n, test_n))

    def test_basic_1d(self):
        """Basic 1D membership test."""
        a_n = np.array([1, 2, 3, 4, 5])
        test_n = np.array([2, 4])
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.isin(a_r, test_r), np.isin(a_n, test_n))

    def test_2d_element(self):
        """Test with 2D element array (should work element-wise)."""
        a_n = np.array([[1, 2], [3, 4]])
        test_n = np.array([2, 3])
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.isin(a_r, test_r), np.isin(a_n, test_n))

    def test_no_overlap(self):
        """Test when no elements match."""
        a_n = np.array([1, 2, 3])
        test_n = np.array([4, 5, 6])
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.isin(a_r, test_r), np.isin(a_n, test_n))

    def test_full_overlap(self):
        """Test when all elements match."""
        a_n = np.array([1, 2, 3])
        test_n = np.array([1, 2, 3, 4, 5])
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.isin(a_r, test_r), np.isin(a_n, test_n))

    def test_empty_element(self):
        """Test with empty element array."""
        a_n = np.array([], dtype=np.int64)
        test_n = np.array([1, 2, 3], dtype=np.int64)
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.isin(a_r, test_r), np.isin(a_n, test_n))

    def test_empty_test(self):
        """Test with empty test array."""
        a_n = np.array([1, 2, 3])
        test_n = np.array([], dtype=np.int64)
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.isin(a_r, test_r), np.isin(a_n, test_n))

    def test_floats(self):
        """Test with float arrays."""
        a_n = np.array([1.5, 2.5, 3.5, 4.5])
        test_n = np.array([2.5, 4.5])
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.isin(a_r, test_r), np.isin(a_n, test_n))

    def test_invert(self):
        """Test with invert flag."""
        a_n = np.array([1, 2, 3, 4, 5])
        test_n = np.array([2, 4])
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.isin(a_r, test_r, invert=True), np.isin(a_n, test_n, invert=True))

    def test_single_element(self):
        """Test with single element arrays."""
        a_n = np.array([5])
        test_n = np.array([5])
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.isin(a_r, test_r), np.isin(a_n, test_n))


class TestIn1d:
    """Test in1d operation (deprecated alias for isin)."""

    def test_basic(self):
        """Basic in1d functionality."""
        a_n = np.array([1, 2, 3, 4, 5])
        test_n = np.array([2, 4])
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.in1d(a_r, test_r), np.in1d(a_n, test_n))

    def test_2d_input(self):
        """Test with 2D input (should be flattened)."""
        a_n = np.array([[1, 2], [3, 4]])
        test_n = np.array([2, 3])
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.in1d(a_r, test_r), np.in1d(a_n, test_n))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, dtype):
        """Test all numeric dtypes."""
        a_n = np.array([1, 2, 3, 4, 5], dtype=dtype)
        test_n = np.array([2, 4], dtype=dtype)
        a_r = rp.asarray(a_n)
        test_r = rp.asarray(test_n)
        assert_eq(rp.in1d(a_r, test_r), np.in1d(a_n, test_n))


class TestBinarySetOps:
    """Test binary set operations: intersect1d, union1d, setdiff1d, setxor1d."""

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, op, dtype):
        """Test all numeric dtypes."""
        a_n = np.array([1, 2, 3, 4, 5], dtype=dtype)
        b_n = np.array([3, 4, 5, 6, 7], dtype=dtype)
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    def test_basic(self, op):
        """Basic operation with overlapping arrays."""
        a_n = np.array([1, 2, 3, 4, 5])
        b_n = np.array([3, 4, 5, 6, 7])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    def test_no_overlap(self, op):
        """Test with disjoint arrays."""
        a_n = np.array([1, 2, 3])
        b_n = np.array([4, 5, 6])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    def test_full_overlap(self, op):
        """Test with identical arrays."""
        a_n = np.array([1, 2, 3])
        b_n = np.array([1, 2, 3])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    def test_with_duplicates(self, op):
        """Test with duplicate values in input arrays."""
        a_n = np.array([1, 1, 2, 2, 3, 3])
        b_n = np.array([2, 2, 3, 3, 4, 4])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    def test_empty_first(self, op):
        """Test with empty first array."""
        a_n = np.array([], dtype=np.int64)
        b_n = np.array([1, 2, 3], dtype=np.int64)
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    def test_empty_second(self, op):
        """Test with empty second array."""
        a_n = np.array([1, 2, 3])
        b_n = np.array([], dtype=np.int64)
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    def test_both_empty(self, op):
        """Test with both arrays empty."""
        a_n = np.array([], dtype=np.int64)
        b_n = np.array([], dtype=np.int64)
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    def test_single_element(self, op):
        """Test with single element arrays."""
        a_n = np.array([5])
        b_n = np.array([5])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    def test_floats(self, op):
        """Test with float arrays."""
        a_n = np.array([1.5, 2.5, 3.5])
        b_n = np.array([2.5, 3.5, 4.5])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))

    @pytest.mark.parametrize("op", BINARY_SET_OPS)
    def test_negative_values(self, op):
        """Test with negative values."""
        a_n = np.array([-3, -1, 0, 1, 2])
        b_n = np.array([-1, 0, 1, 3, 4])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(getattr(rp, op)(a_r, b_r), getattr(np, op)(a_n, b_n))


class TestIntersect1d:
    """Additional specific tests for intersect1d."""

    def test_result_is_sorted(self):
        """Verify result is sorted and unique."""
        a_n = np.array([5, 2, 8, 3, 1])
        b_n = np.array([3, 7, 2, 9])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        result = rp.intersect1d(a_r, b_r)
        expected = np.intersect1d(a_n, b_n)
        assert_eq(result, expected)
        # Verify result is sorted
        assert_eq(result, rp.sort(result))

    def test_subset_relationship(self):
        """Test when one array is a subset of the other."""
        a_n = np.array([1, 2, 3])
        b_n = np.array([1, 2, 3, 4, 5])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(rp.intersect1d(a_r, b_r), np.intersect1d(a_n, b_n))


class TestUnion1d:
    """Additional specific tests for union1d."""

    def test_result_is_sorted_unique(self):
        """Verify result is sorted and unique."""
        a_n = np.array([5, 2, 8, 2])
        b_n = np.array([3, 5, 1, 3])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        result = rp.union1d(a_r, b_r)
        expected = np.union1d(a_n, b_n)
        assert_eq(result, expected)
        # Verify result is sorted and unique
        assert_eq(result, rp.unique(result))
        assert_eq(result, rp.sort(result))


class TestSetdiff1d:
    """Additional specific tests for setdiff1d."""

    def test_result_is_sorted_unique(self):
        """Verify result is sorted and unique."""
        a_n = np.array([5, 2, 8, 3, 2])
        b_n = np.array([3, 7])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        result = rp.setdiff1d(a_r, b_r)
        expected = np.setdiff1d(a_n, b_n)
        assert_eq(result, expected)
        # Verify result is sorted and unique
        assert_eq(result, rp.unique(result))
        assert_eq(result, rp.sort(result))

    def test_complete_removal(self):
        """Test when all elements of first array are in second."""
        a_n = np.array([1, 2, 3])
        b_n = np.array([1, 2, 3, 4, 5])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        result = rp.setdiff1d(a_r, b_r)
        expected = np.setdiff1d(a_n, b_n)
        assert_eq(result, expected)
        assert result.size == 0


class TestSetxor1d:
    """Additional specific tests for setxor1d."""

    def test_result_is_sorted_unique(self):
        """Verify result is sorted and unique."""
        a_n = np.array([5, 2, 8, 3, 2])
        b_n = np.array([3, 7, 2, 9, 7])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        result = rp.setxor1d(a_r, b_r)
        expected = np.setxor1d(a_n, b_n)
        assert_eq(result, expected)
        # Verify result is sorted and unique
        assert_eq(result, rp.unique(result))
        assert_eq(result, rp.sort(result))

    def test_symmetric(self):
        """Verify setxor1d is symmetric (a XOR b == b XOR a)."""
        a_n = np.array([1, 2, 3, 4])
        b_n = np.array([3, 4, 5, 6])
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        assert_eq(rp.setxor1d(a_r, b_r), rp.setxor1d(b_r, a_r))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
