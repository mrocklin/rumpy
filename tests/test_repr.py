"""Tests for __repr__ and __str__ formatting."""

import numpy as np
import rumpy as rp


class TestRepr1D:
    """Test repr for 1D arrays."""

    def test_small_int(self):
        r = rp.arange(5)
        n = np.arange(5)
        assert repr(r) == repr(n)

    def test_small_float(self):
        r = rp.arange(5, dtype="float64")
        n = np.arange(5, dtype=np.float64)
        assert repr(r) == repr(n)

    def test_empty(self):
        r = rp.arange(0)
        n = np.arange(0)
        assert repr(r) == repr(n)


class TestRepr2D:
    """Test repr for 2D arrays."""

    def test_small(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12).reshape(3, 4)
        assert repr(r) == repr(n)

    def test_with_negatives(self):
        r = rp.arange(-6, 6).reshape(3, 4)
        n = np.arange(-6, 6).reshape(3, 4)
        assert repr(r) == repr(n)


class TestRepr3D:
    """Test repr for 3D arrays."""

    def test_small(self):
        r = rp.arange(24).reshape(2, 3, 4)
        n = np.arange(24).reshape(2, 3, 4)
        assert repr(r) == repr(n)


class TestReprTruncation:
    """Test repr truncation for large arrays."""

    def test_1d_truncated(self):
        r = rp.arange(1001)
        n = np.arange(1001)
        assert repr(r) == repr(n)

    def test_2d_truncated(self):
        r = rp.arange(2000).reshape(100, 20)
        n = np.arange(2000).reshape(100, 20)
        assert repr(r) == repr(n)


class TestReprDtypes:
    """Test repr for different dtypes."""

    def test_bool(self):
        # Use explicit dtype since asarray doesn't infer bool yet
        r = rp.asarray([1, 0, 1], dtype="bool")
        n = np.array([True, False, True])
        assert repr(r) == repr(n)

    def test_float32(self):
        r = rp.arange(5, dtype="float32")
        n = np.arange(5, dtype=np.float32)
        assert repr(r) == repr(n)

    def test_int32(self):
        r = rp.arange(5, dtype="int32")
        n = np.arange(5, dtype=np.int32)
        assert repr(r) == repr(n)


class TestStr:
    """Test __str__ formatting."""

    def test_1d(self):
        r = rp.arange(5)
        n = np.arange(5)
        assert str(r) == str(n)

    def test_2d(self):
        r = rp.arange(12).reshape(3, 4)
        n = np.arange(12).reshape(3, 4)
        assert str(r) == str(n)
