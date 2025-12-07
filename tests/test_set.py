"""Tests for set operations - Stream 12."""

import numpy as np
import pytest
import rumpy as rp

from helpers import assert_eq


class TestIsin:
    """Test isin function."""

    def test_basic(self):
        """Basic isin functionality."""
        a = rp.array([1, 2, 3, 4, 5])
        test = rp.array([2, 4])
        r = rp.isin(a, test)

        a_np = np.array([1, 2, 3, 4, 5])
        test_np = np.array([2, 4])
        n = np.isin(a_np, test_np)

        assert_eq(r, n)

    def test_2d_element(self):
        """Test with 2D element array."""
        a = rp.array([[1, 2], [3, 4]])
        test = rp.array([2, 3])
        r = rp.isin(a, test)

        a_np = np.array([[1, 2], [3, 4]])
        test_np = np.array([2, 3])
        n = np.isin(a_np, test_np)

        assert_eq(r, n)

    def test_no_matches(self):
        """Test when no elements match."""
        a = rp.array([1, 2, 3])
        test = rp.array([4, 5, 6])
        r = rp.isin(a, test)

        a_np = np.array([1, 2, 3])
        test_np = np.array([4, 5, 6])
        n = np.isin(a_np, test_np)

        assert_eq(r, n)

    def test_all_match(self):
        """Test when all elements match."""
        a = rp.array([1, 2, 3])
        test = rp.array([1, 2, 3, 4, 5])
        r = rp.isin(a, test)

        a_np = np.array([1, 2, 3])
        test_np = np.array([1, 2, 3, 4, 5])
        n = np.isin(a_np, test_np)

        assert_eq(r, n)

    def test_empty_element(self):
        """Test with empty element array."""
        a = rp.array([])
        test = rp.array([1, 2, 3])
        r = rp.isin(a, test)

        a_np = np.array([])
        test_np = np.array([1, 2, 3])
        n = np.isin(a_np, test_np)

        assert_eq(r, n)

    def test_empty_test(self):
        """Test with empty test array."""
        a = rp.array([1, 2, 3])
        test = rp.array([])
        r = rp.isin(a, test)

        a_np = np.array([1, 2, 3])
        test_np = np.array([])
        n = np.isin(a_np, test_np)

        assert_eq(r, n)

    def test_floats(self):
        """Test with float arrays."""
        a = rp.array([1.5, 2.5, 3.5, 4.5])
        test = rp.array([2.5, 4.5])
        r = rp.isin(a, test)

        a_np = np.array([1.5, 2.5, 3.5, 4.5])
        test_np = np.array([2.5, 4.5])
        n = np.isin(a_np, test_np)

        assert_eq(r, n)

    def test_invert(self):
        """Test with invert flag."""
        a = rp.array([1, 2, 3, 4, 5])
        test = rp.array([2, 4])
        r = rp.isin(a, test, invert=True)

        a_np = np.array([1, 2, 3, 4, 5])
        test_np = np.array([2, 4])
        n = np.isin(a_np, test_np, invert=True)

        assert_eq(r, n)


class TestIn1d:
    """Test in1d function (deprecated alias for isin)."""

    def test_basic(self):
        """Basic in1d functionality."""
        a = rp.array([1, 2, 3, 4, 5])
        test = rp.array([2, 4])
        r = rp.in1d(a, test)

        # in1d flattens input, so compare with numpy's in1d
        a_np = np.array([1, 2, 3, 4, 5])
        test_np = np.array([2, 4])
        n = np.in1d(a_np, test_np)

        assert_eq(r, n)

    def test_2d_input(self):
        """Test with 2D input (should be flattened)."""
        a = rp.array([[1, 2], [3, 4]])
        test = rp.array([2, 3])
        r = rp.in1d(a, test)

        a_np = np.array([[1, 2], [3, 4]])
        test_np = np.array([2, 3])
        n = np.in1d(a_np, test_np)

        assert_eq(r, n)


class TestIntersect1d:
    """Test intersect1d function."""

    def test_basic(self):
        """Basic intersection."""
        a = rp.array([1, 2, 3, 4, 5])
        b = rp.array([3, 4, 5, 6, 7])
        r = rp.intersect1d(a, b)

        a_np = np.array([1, 2, 3, 4, 5])
        b_np = np.array([3, 4, 5, 6, 7])
        n = np.intersect1d(a_np, b_np)

        assert_eq(r, n)

    def test_no_intersection(self):
        """Test when there's no intersection."""
        a = rp.array([1, 2, 3])
        b = rp.array([4, 5, 6])
        r = rp.intersect1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([4, 5, 6])
        n = np.intersect1d(a_np, b_np)

        assert_eq(r, n)

    def test_with_duplicates(self):
        """Test with duplicate values."""
        a = rp.array([1, 2, 2, 3, 3, 3])
        b = rp.array([2, 2, 3, 4])
        r = rp.intersect1d(a, b)

        a_np = np.array([1, 2, 2, 3, 3, 3])
        b_np = np.array([2, 2, 3, 4])
        n = np.intersect1d(a_np, b_np)

        assert_eq(r, n)

    def test_empty_array(self):
        """Test with empty array."""
        a = rp.array([1, 2, 3])
        b = rp.array([])
        r = rp.intersect1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([])
        n = np.intersect1d(a_np, b_np)

        assert_eq(r, n)

    def test_floats(self):
        """Test with float arrays."""
        a = rp.array([1.5, 2.5, 3.5])
        b = rp.array([2.5, 3.5, 4.5])
        r = rp.intersect1d(a, b)

        a_np = np.array([1.5, 2.5, 3.5])
        b_np = np.array([2.5, 3.5, 4.5])
        n = np.intersect1d(a_np, b_np)

        assert_eq(r, n)


class TestUnion1d:
    """Test union1d function."""

    def test_basic(self):
        """Basic union."""
        a = rp.array([1, 2, 3])
        b = rp.array([3, 4, 5])
        r = rp.union1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([3, 4, 5])
        n = np.union1d(a_np, b_np)

        assert_eq(r, n)

    def test_disjoint(self):
        """Test with disjoint arrays."""
        a = rp.array([1, 2, 3])
        b = rp.array([4, 5, 6])
        r = rp.union1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([4, 5, 6])
        n = np.union1d(a_np, b_np)

        assert_eq(r, n)

    def test_identical(self):
        """Test with identical arrays."""
        a = rp.array([1, 2, 3])
        b = rp.array([1, 2, 3])
        r = rp.union1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([1, 2, 3])
        n = np.union1d(a_np, b_np)

        assert_eq(r, n)

    def test_with_duplicates(self):
        """Test with duplicate values."""
        a = rp.array([1, 1, 2, 2])
        b = rp.array([2, 2, 3, 3])
        r = rp.union1d(a, b)

        a_np = np.array([1, 1, 2, 2])
        b_np = np.array([2, 2, 3, 3])
        n = np.union1d(a_np, b_np)

        assert_eq(r, n)

    def test_empty_array(self):
        """Test with empty array."""
        a = rp.array([1, 2, 3])
        b = rp.array([])
        r = rp.union1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([])
        n = np.union1d(a_np, b_np)

        assert_eq(r, n)


class TestSetdiff1d:
    """Test setdiff1d function."""

    def test_basic(self):
        """Basic set difference."""
        a = rp.array([1, 2, 3, 4, 5])
        b = rp.array([3, 4, 5, 6, 7])
        r = rp.setdiff1d(a, b)

        a_np = np.array([1, 2, 3, 4, 5])
        b_np = np.array([3, 4, 5, 6, 7])
        n = np.setdiff1d(a_np, b_np)

        assert_eq(r, n)

    def test_no_difference(self):
        """Test when all elements are in second array."""
        a = rp.array([1, 2, 3])
        b = rp.array([1, 2, 3, 4, 5])
        r = rp.setdiff1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([1, 2, 3, 4, 5])
        n = np.setdiff1d(a_np, b_np)

        assert_eq(r, n)

    def test_all_different(self):
        """Test when arrays are disjoint."""
        a = rp.array([1, 2, 3])
        b = rp.array([4, 5, 6])
        r = rp.setdiff1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([4, 5, 6])
        n = np.setdiff1d(a_np, b_np)

        assert_eq(r, n)

    def test_with_duplicates(self):
        """Test with duplicate values."""
        a = rp.array([1, 2, 2, 3, 3, 3])
        b = rp.array([2, 4])
        r = rp.setdiff1d(a, b)

        a_np = np.array([1, 2, 2, 3, 3, 3])
        b_np = np.array([2, 4])
        n = np.setdiff1d(a_np, b_np)

        assert_eq(r, n)

    def test_empty_first(self):
        """Test with empty first array."""
        a = rp.array([])
        b = rp.array([1, 2, 3])
        r = rp.setdiff1d(a, b)

        a_np = np.array([])
        b_np = np.array([1, 2, 3])
        n = np.setdiff1d(a_np, b_np)

        assert_eq(r, n)

    def test_empty_second(self):
        """Test with empty second array."""
        a = rp.array([1, 2, 3])
        b = rp.array([])
        r = rp.setdiff1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([])
        n = np.setdiff1d(a_np, b_np)

        assert_eq(r, n)


class TestSetxor1d:
    """Test setxor1d function."""

    def test_basic(self):
        """Basic symmetric difference."""
        a = rp.array([1, 2, 3, 4])
        b = rp.array([3, 4, 5, 6])
        r = rp.setxor1d(a, b)

        a_np = np.array([1, 2, 3, 4])
        b_np = np.array([3, 4, 5, 6])
        n = np.setxor1d(a_np, b_np)

        assert_eq(r, n)

    def test_disjoint(self):
        """Test with disjoint arrays."""
        a = rp.array([1, 2, 3])
        b = rp.array([4, 5, 6])
        r = rp.setxor1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([4, 5, 6])
        n = np.setxor1d(a_np, b_np)

        assert_eq(r, n)

    def test_identical(self):
        """Test with identical arrays."""
        a = rp.array([1, 2, 3])
        b = rp.array([1, 2, 3])
        r = rp.setxor1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([1, 2, 3])
        n = np.setxor1d(a_np, b_np)

        assert_eq(r, n)

    def test_with_duplicates(self):
        """Test with duplicate values."""
        a = rp.array([1, 1, 2, 2, 3])
        b = rp.array([2, 3, 3, 4, 4])
        r = rp.setxor1d(a, b)

        a_np = np.array([1, 1, 2, 2, 3])
        b_np = np.array([2, 3, 3, 4, 4])
        n = np.setxor1d(a_np, b_np)

        assert_eq(r, n)

    def test_empty_array(self):
        """Test with empty array."""
        a = rp.array([1, 2, 3])
        b = rp.array([])
        r = rp.setxor1d(a, b)

        a_np = np.array([1, 2, 3])
        b_np = np.array([])
        n = np.setxor1d(a_np, b_np)

        assert_eq(r, n)

    def test_both_empty(self):
        """Test with both arrays empty."""
        a = rp.array([])
        b = rp.array([])
        r = rp.setxor1d(a, b)

        a_np = np.array([])
        b_np = np.array([])
        n = np.setxor1d(a_np, b_np)

        assert_eq(r, n)
