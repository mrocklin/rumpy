"""Tests for Stream 2: Binary Math Operations."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestArctan2:
    """Tests for arctan2(y, x)."""

    def test_basic(self):
        y = rp.array([1.0, -1.0, 1.0, -1.0])
        x = rp.array([1.0, 1.0, -1.0, -1.0])
        ny = np.array([1.0, -1.0, 1.0, -1.0])
        nx = np.array([1.0, 1.0, -1.0, -1.0])
        assert_eq(rp.arctan2(y, x), np.arctan2(ny, nx))

    def test_broadcast(self):
        y = rp.arange(1.0, 5.0)
        x = rp.array([2.0])
        ny = np.arange(1.0, 5.0)
        nx = np.array([2.0])
        assert_eq(rp.arctan2(y, x), np.arctan2(ny, nx))

    def test_zeros(self):
        y = rp.array([0.0, 1.0, 0.0])
        x = rp.array([1.0, 0.0, 0.0])
        ny = np.array([0.0, 1.0, 0.0])
        nx = np.array([1.0, 0.0, 0.0])
        assert_eq(rp.arctan2(y, x), np.arctan2(ny, nx))


class TestHypot:
    """Tests for hypot(x1, x2) = sqrt(x1^2 + x2^2)."""

    def test_basic(self):
        x1 = rp.array([3.0, 5.0, 0.0])
        x2 = rp.array([4.0, 12.0, 0.0])
        n1 = np.array([3.0, 5.0, 0.0])
        n2 = np.array([4.0, 12.0, 0.0])
        assert_eq(rp.hypot(x1, x2), np.hypot(n1, n2))

    def test_broadcast(self):
        x1 = rp.arange(1.0, 5.0)
        x2 = rp.array([2.0])
        n1 = np.arange(1.0, 5.0)
        n2 = np.array([2.0])
        assert_eq(rp.hypot(x1, x2), np.hypot(n1, n2))


class TestFmaxFmin:
    """Tests for fmax and fmin (ignore NaN)."""

    def test_fmax_basic(self):
        x1 = rp.array([1.0, 5.0, 3.0])
        x2 = rp.array([2.0, 3.0, 4.0])
        n1 = np.array([1.0, 5.0, 3.0])
        n2 = np.array([2.0, 3.0, 4.0])
        assert_eq(rp.fmax(x1, x2), np.fmax(n1, n2))

    def test_fmin_basic(self):
        x1 = rp.array([1.0, 5.0, 3.0])
        x2 = rp.array([2.0, 3.0, 4.0])
        n1 = np.array([1.0, 5.0, 3.0])
        n2 = np.array([2.0, 3.0, 4.0])
        assert_eq(rp.fmin(x1, x2), np.fmin(n1, n2))

    def test_fmax_with_nan(self):
        x1 = rp.array([1.0, np.nan, 3.0])
        x2 = rp.array([2.0, 3.0, np.nan])
        n1 = np.array([1.0, np.nan, 3.0])
        n2 = np.array([2.0, 3.0, np.nan])
        assert_eq(rp.fmax(x1, x2), np.fmax(n1, n2))

    def test_fmin_with_nan(self):
        x1 = rp.array([1.0, np.nan, 3.0])
        x2 = rp.array([2.0, 3.0, np.nan])
        n1 = np.array([1.0, np.nan, 3.0])
        n2 = np.array([2.0, 3.0, np.nan])
        assert_eq(rp.fmin(x1, x2), np.fmin(n1, n2))


class TestCopysign:
    """Tests for copysign(x1, x2)."""

    def test_basic(self):
        x1 = rp.array([1.0, -2.0, 3.0, -4.0])
        x2 = rp.array([1.0, 1.0, -1.0, -1.0])
        n1 = np.array([1.0, -2.0, 3.0, -4.0])
        n2 = np.array([1.0, 1.0, -1.0, -1.0])
        assert_eq(rp.copysign(x1, x2), np.copysign(n1, n2))

    def test_zero(self):
        x1 = rp.array([0.0, 1.0])
        x2 = rp.array([-1.0, 0.0])
        n1 = np.array([0.0, 1.0])
        n2 = np.array([-1.0, 0.0])
        assert_eq(rp.copysign(x1, x2), np.copysign(n1, n2))


class TestLogaddexp:
    """Tests for logaddexp = log(exp(x1) + exp(x2))."""

    def test_basic(self):
        x1 = rp.array([1.0, 2.0, 3.0])
        x2 = rp.array([2.0, 3.0, 4.0])
        n1 = np.array([1.0, 2.0, 3.0])
        n2 = np.array([2.0, 3.0, 4.0])
        assert_eq(rp.logaddexp(x1, x2), np.logaddexp(n1, n2))

    def test_negative(self):
        x1 = rp.array([-1.0, -10.0])
        x2 = rp.array([-2.0, -11.0])
        n1 = np.array([-1.0, -10.0])
        n2 = np.array([-2.0, -11.0])
        assert_eq(rp.logaddexp(x1, x2), np.logaddexp(n1, n2))


class TestLogaddexp2:
    """Tests for logaddexp2 = log2(2^x1 + 2^x2)."""

    def test_basic(self):
        x1 = rp.array([1.0, 2.0, 3.0])
        x2 = rp.array([2.0, 3.0, 4.0])
        n1 = np.array([1.0, 2.0, 3.0])
        n2 = np.array([2.0, 3.0, 4.0])
        assert_eq(rp.logaddexp2(x1, x2), np.logaddexp2(n1, n2))


class TestNextafter:
    """Tests for nextafter(x1, x2)."""

    def test_basic(self):
        x1 = rp.array([1.0, 1.0, 1.0])
        x2 = rp.array([2.0, 0.0, 1.0])
        n1 = np.array([1.0, 1.0, 1.0])
        n2 = np.array([2.0, 0.0, 1.0])
        assert_eq(rp.nextafter(x1, x2), np.nextafter(n1, n2))


class TestDeg2Rad:
    """Tests for deg2rad."""

    def test_basic(self):
        r = rp.array([0.0, 90.0, 180.0, 270.0, 360.0])
        n = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
        assert_eq(rp.deg2rad(r), np.deg2rad(n))

    def test_scalar(self):
        assert abs(rp.deg2rad(180.0) - np.pi) < 1e-10


class TestRad2Deg:
    """Tests for rad2deg."""

    def test_basic(self):
        r = rp.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        n = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        assert_eq(rp.rad2deg(r), np.rad2deg(n))

    def test_scalar(self):
        assert abs(rp.rad2deg(np.pi) - 180.0) < 1e-10


class TestFloat32:
    """Tests for float32 dtype."""

    def test_arctan2_float32(self):
        y = rp.array([1.0, -1.0], dtype="float32")
        x = rp.array([1.0, 1.0], dtype="float32")
        ny = np.array([1.0, -1.0], dtype=np.float32)
        nx = np.array([1.0, 1.0], dtype=np.float32)
        assert_eq(rp.arctan2(y, x), np.arctan2(ny, nx))

    def test_hypot_float32(self):
        x1 = rp.array([3.0, 5.0], dtype="float32")
        x2 = rp.array([4.0, 12.0], dtype="float32")
        n1 = np.array([3.0, 5.0], dtype=np.float32)
        n2 = np.array([4.0, 12.0], dtype=np.float32)
        assert_eq(rp.hypot(x1, x2), np.hypot(n1, n2))

    def test_fmax_float32(self):
        x1 = rp.array([1.0, np.nan], dtype="float32")
        x2 = rp.array([2.0, 3.0], dtype="float32")
        n1 = np.array([1.0, np.nan], dtype=np.float32)
        n2 = np.array([2.0, 3.0], dtype=np.float32)
        assert_eq(rp.fmax(x1, x2), np.fmax(n1, n2))
