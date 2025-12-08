"""Tests for known issues and unsupported features.

These tests document gaps in rumpy's numpy compatibility.
They are marked xfail so they don't block CI but will alert
us when issues are fixed.
"""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestNegativeAxisND:
    """Negative axis fails on 3D+ arrays."""

    @pytest.mark.xfail(reason="negative axis broken on 3D arrays", strict=True)
    @pytest.mark.parametrize("axis", [-1, -2, -3])
    def test_sum_negative_axis_3d(self, axis):
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)
        assert_eq(rp.sum(r, axis=axis), np.sum(n, axis=axis))

    @pytest.mark.xfail(reason="negative axis broken on 3D arrays", strict=True)
    @pytest.mark.parametrize("axis", [-1, -2, -3])
    def test_mean_negative_axis_3d(self, axis):
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)
        assert_eq(rp.mean(r, axis=axis), np.mean(n, axis=axis))

    @pytest.mark.xfail(reason="negative axis broken on 3D arrays", strict=True)
    def test_argmax_negative_axis_3d(self):
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)
        assert_eq(rp.argmax(r, axis=-1), np.argmax(n, axis=-1))


class TestTupleAxis:
    """Tuple of axes not supported."""

    @pytest.mark.xfail(reason="tuple axis not supported", strict=True)
    def test_sum_tuple_axis(self):
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)
        assert_eq(rp.sum(r, axis=(0, 2)), np.sum(n, axis=(0, 2)))

    @pytest.mark.xfail(reason="tuple axis not supported", strict=True)
    def test_mean_tuple_axis(self):
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)
        assert_eq(rp.mean(r, axis=(0, 1)), np.mean(n, axis=(0, 1)))


class TestSmallIntDtypes:
    """int8, int16, uint16 not supported."""

    @pytest.mark.xfail(reason="int8 not supported", strict=True)
    def test_int8_creation(self):
        n = np.array([1, 2, 3], dtype=np.int8)
        r = rp.asarray(n)
        assert_eq(r, n)

    @pytest.mark.xfail(reason="int16 not supported", strict=True)
    def test_int16_creation(self):
        n = np.array([1, 2, 3], dtype=np.int16)
        r = rp.asarray(n)
        assert_eq(r, n)

    @pytest.mark.xfail(reason="uint16 not supported", strict=True)
    def test_uint16_creation(self):
        n = np.array([1, 2, 3], dtype=np.uint16)
        r = rp.asarray(n)
        assert_eq(r, n)


class TestFloat16:
    """float16 import from numpy not supported (creation works)."""

    @pytest.mark.xfail(reason="float16 asarray from numpy not supported", strict=True)
    def test_float16_from_numpy(self):
        n = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        r = rp.asarray(n)
        assert_eq(r, n)
