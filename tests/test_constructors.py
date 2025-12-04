"""Tests for array constructors."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestLinspace:
    """Test linspace constructor."""

    def test_linspace_basic(self):
        r = rp.linspace(0, 10, 5)
        n = np.linspace(0, 10, 5)
        assert_eq(r, n)

    def test_linspace_default_num(self):
        r = rp.linspace(0, 1)
        n = np.linspace(0, 1)  # default 50 points
        assert_eq(r, n)

    def test_linspace_single_point(self):
        r = rp.linspace(5, 5, 1)
        n = np.linspace(5, 5, 1)
        assert_eq(r, n)

    def test_linspace_negative(self):
        r = rp.linspace(-5, 5, 11)
        n = np.linspace(-5, 5, 11)
        assert_eq(r, n)

    def test_linspace_reverse(self):
        r = rp.linspace(10, 0, 5)
        n = np.linspace(10, 0, 5)
        assert_eq(r, n)

    def test_linspace_empty(self):
        r = rp.linspace(0, 10, 0)
        n = np.linspace(0, 10, 0)
        assert_eq(r, n)

    def test_linspace_two_points(self):
        r = rp.linspace(0, 10, 2)
        n = np.linspace(0, 10, 2)
        assert_eq(r, n)


class TestEye:
    """Test eye constructor."""

    def test_eye_basic(self):
        r = rp.eye(3)
        n = np.eye(3)
        assert_eq(r, n)

    def test_eye_1x1(self):
        r = rp.eye(1)
        n = np.eye(1)
        assert_eq(r, n)

    def test_eye_empty(self):
        r = rp.eye(0)
        n = np.eye(0)
        assert_eq(r, n)

    def test_eye_large(self):
        r = rp.eye(10)
        n = np.eye(10)
        assert_eq(r, n)

    def test_eye_int_dtype(self):
        r = rp.eye(3, dtype="int64")
        n = np.eye(3, dtype=np.int64)
        assert_eq(r, n)


class TestFull:
    """Test full constructor."""

    def test_full_1d(self):
        r = rp.full(5, 3.14)
        n = np.full(5, 3.14)
        assert_eq(r, n)

    def test_full_2d(self):
        r = rp.full((3, 4), 2.5)
        n = np.full((3, 4), 2.5)
        assert_eq(r, n)

    def test_full_3d(self):
        r = rp.full((2, 3, 4), -1.0)
        n = np.full((2, 3, 4), -1.0)
        assert_eq(r, n)

    def test_full_zero(self):
        r = rp.full(5, 0.0)
        n = np.full(5, 0.0)
        assert_eq(r, n)

    def test_full_negative(self):
        r = rp.full((3, 3), -42.0)
        n = np.full((3, 3), -42.0)
        assert_eq(r, n)

    def test_full_empty(self):
        r = rp.full(0, 5.0)
        n = np.full(0, 5.0)
        assert_eq(r, n)

    def test_full_int_dtype(self):
        r = rp.full((3, 3), 7.0, dtype="int64")
        n = np.full((3, 3), 7, dtype=np.int64)
        assert_eq(r, n)
