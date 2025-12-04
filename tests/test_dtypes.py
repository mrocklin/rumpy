"""Tests for dtype system - new dtypes and parametric types."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestUnsignedIntegers:
    """Test unsigned integer dtypes."""

    def test_uint8_zeros(self):
        r = rp.zeros(10, dtype="uint8")
        n = np.zeros(10, dtype=np.uint8)
        assert_eq(r, n)

    def test_uint8_ones(self):
        r = rp.ones(10, dtype="uint8")
        n = np.ones(10, dtype=np.uint8)
        assert_eq(r, n)

    def test_uint8_arange(self):
        r = rp.arange(10, dtype="uint8")
        n = np.arange(10, dtype=np.uint8)
        assert_eq(r, n)

    def test_uint8_itemsize(self):
        r = rp.zeros(10, dtype="uint8")
        assert r.itemsize == 1

    def test_uint32_zeros(self):
        r = rp.zeros(10, dtype="uint32")
        n = np.zeros(10, dtype=np.uint32)
        assert_eq(r, n)

    def test_uint32_ones(self):
        r = rp.ones(10, dtype="uint32")
        n = np.ones(10, dtype=np.uint32)
        assert_eq(r, n)

    def test_uint32_arange(self):
        r = rp.arange(100, dtype="uint32")
        n = np.arange(100, dtype=np.uint32)
        assert_eq(r, n)

    def test_uint32_itemsize(self):
        r = rp.zeros(10, dtype="uint32")
        assert r.itemsize == 4

    def test_uint64_zeros(self):
        r = rp.zeros(10, dtype="uint64")
        n = np.zeros(10, dtype=np.uint64)
        assert_eq(r, n)

    def test_uint64_ones(self):
        r = rp.ones(10, dtype="uint64")
        n = np.ones(10, dtype=np.uint64)
        assert_eq(r, n)

    def test_uint64_arange(self):
        r = rp.arange(100, dtype="uint64")
        n = np.arange(100, dtype=np.uint64)
        assert_eq(r, n)

    def test_uint64_itemsize(self):
        r = rp.zeros(10, dtype="uint64")
        assert r.itemsize == 8


class TestUnsignedInterop:
    """Test numpy interop with unsigned types."""

    def test_numpy_uint8_to_rumpy(self):
        n = np.array([1, 2, 3, 255], dtype=np.uint8)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_numpy_uint32_to_rumpy(self):
        n = np.arange(10, dtype=np.uint32)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_numpy_uint64_to_rumpy(self):
        n = np.arange(10, dtype=np.uint64)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_rumpy_uint8_to_numpy(self):
        r = rp.arange(10, dtype="uint8")
        n = np.asarray(r)
        assert n.dtype == np.uint8
        assert_eq(r, n)

    def test_rumpy_uint32_to_numpy(self):
        r = rp.arange(10, dtype="uint32")
        n = np.asarray(r)
        assert n.dtype == np.uint32
        assert_eq(r, n)

    def test_rumpy_uint64_to_numpy(self):
        r = rp.arange(10, dtype="uint64")
        n = np.asarray(r)
        assert n.dtype == np.uint64
        assert_eq(r, n)


class TestUnsignedOperations:
    """Test operations on unsigned types."""

    def test_uint8_add(self):
        r1 = rp.arange(5, dtype="uint8")
        r2 = rp.ones(5, dtype="uint8")
        n1 = np.arange(5, dtype=np.uint8)
        n2 = np.ones(5, dtype=np.uint8)
        assert_eq(r1 + r2, n1 + n2)

    def test_uint32_mul(self):
        r = rp.arange(5, dtype="uint32")
        n = np.arange(5, dtype=np.uint32)
        assert_eq(r * r, n * n)

    def test_uint64_sum(self):
        r = rp.arange(10, dtype="uint64")
        n = np.arange(10, dtype=np.uint64)
        assert r.sum() == n.sum()

    def test_uint8_2d(self):
        r = rp.ones((3, 4), dtype="uint8")
        n = np.ones((3, 4), dtype=np.uint8)
        assert_eq(r, n)

    def test_uint32_reshape(self):
        r = rp.arange(12, dtype="uint32").reshape(3, 4)
        n = np.arange(12, dtype=np.uint32).reshape(3, 4)
        assert_eq(r, n)


class TestDateTime64:
    """Test datetime64 parametric dtype."""

    def test_datetime64_ns_zeros(self):
        r = rp.zeros(5, dtype="datetime64[ns]")
        n = np.zeros(5, dtype="datetime64[ns]")
        assert r.dtype == "datetime64[ns]"
        assert r.itemsize == 8
        # NumPy will interpret our array as datetime64 via __array_interface__
        assert_eq(r, n)

    def test_datetime64_us_zeros(self):
        r = rp.zeros(5, dtype="datetime64[us]")
        n = np.zeros(5, dtype="datetime64[us]")
        assert r.dtype == "datetime64[us]"
        assert_eq(r, n)

    def test_datetime64_ms_zeros(self):
        r = rp.zeros(5, dtype="datetime64[ms]")
        n = np.zeros(5, dtype="datetime64[ms]")
        assert r.dtype == "datetime64[ms]"
        assert_eq(r, n)

    def test_datetime64_s_zeros(self):
        r = rp.zeros(5, dtype="datetime64[s]")
        n = np.zeros(5, dtype="datetime64[s]")
        assert r.dtype == "datetime64[s]"
        assert_eq(r, n)

    def test_datetime64_different_units_not_equal(self):
        """Different time units should be different dtypes."""
        r_ns = rp.zeros(1, dtype="datetime64[ns]")
        r_us = rp.zeros(1, dtype="datetime64[us]")
        assert r_ns.dtype != r_us.dtype

    def test_datetime64_same_unit_equal(self):
        """Same time unit should be equal dtype."""
        r1 = rp.zeros(1, dtype="datetime64[ns]")
        r2 = rp.ones(1, dtype="datetime64[ns]")
        assert r1.dtype == r2.dtype


class TestDateTime64Interop:
    """Test numpy interop with datetime64."""

    def test_numpy_datetime64_ns_to_rumpy(self):
        n = np.array(['2024-01-01', '2024-01-02'], dtype="datetime64[ns]")
        r = rp.asarray(n)
        assert r.dtype == "datetime64[ns]"
        # Values should match when converted back
        assert_eq(r, n)

    def test_numpy_datetime64_s_to_rumpy(self):
        n = np.array(['2024-01-01', '2024-01-02'], dtype="datetime64[s]")
        r = rp.asarray(n)
        assert r.dtype == "datetime64[s]"
        assert_eq(r, n)

    def test_rumpy_datetime64_to_numpy(self):
        """Round-trip: rumpy -> numpy -> compare."""
        r = rp.zeros(3, dtype="datetime64[ns]")
        n = np.asarray(r)
        # NumPy should see it as datetime64[ns]
        assert n.dtype == np.dtype("datetime64[ns]")

    def test_datetime64_2d(self):
        r = rp.zeros((2, 3), dtype="datetime64[ns]")
        n = np.zeros((2, 3), dtype="datetime64[ns]")
        assert r.shape == (2, 3)
        assert_eq(r, n)
