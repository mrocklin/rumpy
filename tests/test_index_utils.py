"""Tests for index utility functions (Stream 26)."""

import numpy as np
import pytest
import rumpy as rp

from helpers import assert_eq


class TestUnravelIndex:
    """Tests for unravel_index - convert flat index to multi-index."""

    def test_basic_2d(self):
        """Basic 2D case (scalar input returns scalars)."""
        result = rp.unravel_index(22, (7, 6))
        expected = np.unravel_index(22, (7, 6))
        assert result[0] == int(expected[0])
        assert result[1] == int(expected[1])

    def test_multiple_indices(self):
        """Multiple indices at once."""
        result = rp.unravel_index([22, 41, 37], (7, 6))
        expected = np.unravel_index([22, 41, 37], (7, 6))
        for r, e in zip(result, expected):
            assert_eq(rp.asarray(r), e)

    def test_3d_shape(self):
        """3D shape (scalar input returns scalars)."""
        result = rp.unravel_index(100, (6, 7, 8))
        expected = np.unravel_index(100, (6, 7, 8))
        for r, e in zip(result, expected):
            assert r == int(e)

    def test_order_f(self):
        """Fortran order (scalar input returns scalars)."""
        result = rp.unravel_index(22, (7, 6), order="F")
        expected = np.unravel_index(22, (7, 6), order="F")
        for r, e in zip(result, expected):
            assert r == int(e)

    def test_array_input(self):
        """Array input."""
        indices = rp.asarray([0, 5, 10, 15])
        result = rp.unravel_index(indices, (4, 5))
        expected = np.unravel_index([0, 5, 10, 15], (4, 5))
        for r, e in zip(result, expected):
            assert_eq(rp.asarray(r), e)


class TestRavelMultiIndex:
    """Tests for ravel_multi_index - convert multi-index to flat index."""

    def test_basic(self):
        """Basic case."""
        result = rp.ravel_multi_index([[3, 6, 6], [4, 5, 1]], (7, 6))
        expected = np.ravel_multi_index([[3, 6, 6], [4, 5, 1]], (7, 6))
        assert_eq(result, expected)

    def test_mode_clip(self):
        """Clip mode for out-of-bounds."""
        result = rp.ravel_multi_index([[3, 8, 6], [4, 5, 1]], (7, 6), mode="clip")
        expected = np.ravel_multi_index([[3, 8, 6], [4, 5, 1]], (7, 6), mode="clip")
        assert_eq(result, expected)

    def test_mode_wrap(self):
        """Wrap mode for out-of-bounds."""
        result = rp.ravel_multi_index([[3, 6, 6], [4, 5, 1]], (4, 4), mode="wrap")
        expected = np.ravel_multi_index([[3, 6, 6], [4, 5, 1]], (4, 4), mode="wrap")
        assert_eq(result, expected)

    def test_order_f(self):
        """Fortran order."""
        result = rp.ravel_multi_index([[3], [4]], (7, 6), order="F")
        expected = np.ravel_multi_index([[3], [4]], (7, 6), order="F")
        assert_eq(result, expected)


class TestDiagIndices:
    """Tests for diag_indices - return indices for main diagonal."""

    def test_basic(self):
        """Basic 2D case."""
        result = rp.diag_indices(4)
        expected = np.diag_indices(4)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_ndim_3(self):
        """3D case with ndim parameter."""
        result = rp.diag_indices(2, ndim=3)
        expected = np.diag_indices(2, ndim=3)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_large(self):
        """Larger array."""
        result = rp.diag_indices(100)
        expected = np.diag_indices(100)
        for r, e in zip(result, expected):
            assert_eq(r, e)


class TestDiagIndicesFrom:
    """Tests for diag_indices_from - diagonal indices from array shape."""

    def test_square_2d(self):
        """Square 2D array."""
        a = rp.zeros((4, 4))
        n = np.zeros((4, 4))
        result = rp.diag_indices_from(a)
        expected = np.diag_indices_from(n)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_3d(self):
        """3D array with equal dimensions."""
        a = rp.zeros((3, 3, 3))
        n = np.zeros((3, 3, 3))
        result = rp.diag_indices_from(a)
        expected = np.diag_indices_from(n)
        for r, e in zip(result, expected):
            assert_eq(r, e)


class TestTrilIndices:
    """Tests for tril_indices - lower triangle indices."""

    def test_basic(self):
        """Basic case."""
        result = rp.tril_indices(4)
        expected = np.tril_indices(4)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_k_negative(self):
        """Below main diagonal."""
        result = rp.tril_indices(4, k=-1)
        expected = np.tril_indices(4, k=-1)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_k_positive(self):
        """Above main diagonal."""
        result = rp.tril_indices(4, k=1)
        expected = np.tril_indices(4, k=1)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_rectangular(self):
        """Rectangular matrix (n != m)."""
        result = rp.tril_indices(3, m=4)
        expected = np.tril_indices(3, m=4)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_m_greater_than_n(self):
        """m > n case."""
        result = rp.tril_indices(3, m=5)
        expected = np.tril_indices(3, m=5)
        for r, e in zip(result, expected):
            assert_eq(r, e)


class TestTriuIndices:
    """Tests for triu_indices - upper triangle indices."""

    def test_basic(self):
        """Basic case."""
        result = rp.triu_indices(4)
        expected = np.triu_indices(4)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_k_positive(self):
        """Above main diagonal."""
        result = rp.triu_indices(4, k=1)
        expected = np.triu_indices(4, k=1)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_rectangular(self):
        """Rectangular matrix."""
        result = rp.triu_indices(3, m=4)
        expected = np.triu_indices(3, m=4)
        for r, e in zip(result, expected):
            assert_eq(r, e)


class TestTrilIndicesFrom:
    """Tests for tril_indices_from."""

    def test_square(self):
        """Square array."""
        a = rp.zeros((4, 4))
        n = np.zeros((4, 4))
        result = rp.tril_indices_from(a)
        expected = np.tril_indices_from(n)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_rectangular(self):
        """Rectangular array."""
        a = rp.zeros((3, 5))
        n = np.zeros((3, 5))
        result = rp.tril_indices_from(a)
        expected = np.tril_indices_from(n)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_with_k(self):
        """With k offset."""
        a = rp.zeros((4, 4))
        n = np.zeros((4, 4))
        result = rp.tril_indices_from(a, k=1)
        expected = np.tril_indices_from(n, k=1)
        for r, e in zip(result, expected):
            assert_eq(r, e)


class TestTriuIndicesFrom:
    """Tests for triu_indices_from."""

    def test_square(self):
        """Square array."""
        a = rp.zeros((4, 4))
        n = np.zeros((4, 4))
        result = rp.triu_indices_from(a)
        expected = np.triu_indices_from(n)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_rectangular(self):
        """Rectangular array."""
        a = rp.zeros((3, 5))
        n = np.zeros((3, 5))
        result = rp.triu_indices_from(a)
        expected = np.triu_indices_from(n)
        for r, e in zip(result, expected):
            assert_eq(r, e)


class TestMaskIndices:
    """Tests for mask_indices - indices from mask function."""

    def test_triu(self):
        """Using triu as mask function."""
        result = rp.mask_indices(4, rp.triu)
        expected = np.mask_indices(4, np.triu)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_triu_with_k(self):
        """Using triu with k parameter."""
        result = rp.mask_indices(4, rp.triu, k=1)
        expected = np.mask_indices(4, np.triu, k=1)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_tril(self):
        """Using tril as mask function."""
        result = rp.mask_indices(4, rp.tril)
        expected = np.mask_indices(4, np.tril)
        for r, e in zip(result, expected):
            assert_eq(r, e)


class TestDigitize:
    """Tests for digitize - bin indices for values."""

    def test_basic(self):
        """Basic case."""
        x = rp.asarray([0.2, 6.4, 3.0, 1.6])
        bins = rp.asarray([0.0, 1.0, 2.5, 4.0, 10.0])
        n_x = np.array([0.2, 6.4, 3.0, 1.6])
        n_bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        result = rp.digitize(x, bins)
        expected = np.digitize(n_x, n_bins)
        assert_eq(result, expected)

    def test_right(self):
        """With right=True."""
        x = rp.asarray([0.2, 6.4, 3.0, 1.6])
        bins = rp.asarray([0.0, 1.0, 2.5, 4.0, 10.0])
        n_x = np.array([0.2, 6.4, 3.0, 1.6])
        n_bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        result = rp.digitize(x, bins, right=True)
        expected = np.digitize(n_x, n_bins, right=True)
        assert_eq(result, expected)

    def test_decreasing_bins(self):
        """Decreasing bin sequence."""
        x = rp.asarray([0.2, 6.4, 3.0, 1.6])
        bins = rp.asarray([10.0, 4.0, 2.5, 1.0, 0.0])
        n_x = np.array([0.2, 6.4, 3.0, 1.6])
        n_bins = np.array([10.0, 4.0, 2.5, 1.0, 0.0])
        result = rp.digitize(x, bins)
        expected = np.digitize(n_x, n_bins)
        assert_eq(result, expected)

    def test_edge_cases(self):
        """Values at bin edges."""
        x = rp.asarray([0.0, 1.0, 2.5, 4.0, 10.0])
        bins = rp.asarray([0.0, 1.0, 2.5, 4.0, 10.0])
        n_x = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        n_bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        result = rp.digitize(x, bins)
        expected = np.digitize(n_x, n_bins)
        assert_eq(result, expected)


class TestPackbits:
    """Tests for packbits - pack binary array to uint8."""

    def test_1d(self):
        """1D array packing."""
        a = rp.asarray([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
        n = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1], dtype=np.uint8)
        result = rp.packbits(a)
        expected = np.packbits(n)
        assert_eq(result, expected)

    def test_2d_default(self):
        """2D array default (axis=-1)."""
        a = rp.asarray([[1, 0, 1], [0, 1, 0]])
        n = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        result = rp.packbits(a)
        expected = np.packbits(n)
        assert_eq(result, expected)

    def test_axis_0(self):
        """Pack along axis 0."""
        a = rp.asarray([[1, 0, 1], [0, 1, 0]])
        n = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        result = rp.packbits(a, axis=0)
        expected = np.packbits(n, axis=0)
        assert_eq(result, expected)

    def test_axis_1(self):
        """Pack along axis 1."""
        a = rp.asarray([[1, 0, 1], [0, 1, 0]])
        n = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        result = rp.packbits(a, axis=1)
        expected = np.packbits(n, axis=1)
        assert_eq(result, expected)

    def test_axis_none(self):
        """Pack flattened array."""
        a = rp.asarray([[1, 0, 1], [0, 1, 0]])
        n = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        result = rp.packbits(a, axis=None)
        expected = np.packbits(n, axis=None)
        assert_eq(result, expected)

    def test_bitorder_little(self):
        """Little-endian bit order."""
        a = rp.asarray([1, 0, 0, 0, 0, 0, 1, 0])
        n = np.array([1, 0, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
        result = rp.packbits(a, bitorder="little")
        expected = np.packbits(n, bitorder="little")
        assert_eq(result, expected)


class TestUnpackbits:
    """Tests for unpackbits - unpack uint8 to binary array."""

    def test_1d(self):
        """1D array unpacking."""
        a = rp.asarray([202, 192], dtype="uint8")
        n = np.array([202, 192], dtype=np.uint8)
        result = rp.unpackbits(a)
        expected = np.unpackbits(n)
        assert_eq(result, expected)

    def test_2d_axis1(self):
        """2D array along axis 1."""
        a = rp.asarray([[202], [96]], dtype="uint8")
        n = np.array([[202], [96]], dtype=np.uint8)
        result = rp.unpackbits(a, axis=1)
        expected = np.unpackbits(n, axis=1)
        assert_eq(result, expected)

    def test_count(self):
        """Limit output with count parameter."""
        a = rp.asarray([[202], [96]], dtype="uint8")
        n = np.array([[202], [96]], dtype=np.uint8)
        result = rp.unpackbits(a, axis=1, count=4)
        expected = np.unpackbits(n, axis=1, count=4)
        assert_eq(result, expected)

    def test_bitorder_little(self):
        """Little-endian bit order."""
        a = rp.asarray([2], dtype="uint8")
        n = np.array([2], dtype=np.uint8)
        result = rp.unpackbits(a, bitorder="little")
        expected = np.unpackbits(n, bitorder="little")
        assert_eq(result, expected)

    def test_bitorder_big(self):
        """Big-endian bit order (default)."""
        a = rp.asarray([2], dtype="uint8")
        n = np.array([2], dtype=np.uint8)
        result = rp.unpackbits(a, bitorder="big")
        expected = np.unpackbits(n, bitorder="big")
        assert_eq(result, expected)


class TestRoundTrip:
    """Tests for packbits/unpackbits round-trip."""

    def test_pack_unpack(self):
        """Pack then unpack should preserve data."""
        original = rp.asarray([1, 0, 1, 1, 0, 0, 1, 0])
        n_original = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        packed = rp.packbits(original)
        unpacked = rp.unpackbits(packed)
        n_packed = np.packbits(n_original)
        n_unpacked = np.unpackbits(n_packed)
        assert_eq(unpacked, n_unpacked)
