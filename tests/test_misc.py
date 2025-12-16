"""Tests for Stream 32: Miscellaneous Operations.

Functions:
- resize: resize array with repetition
- unstack: split along axis into tuple of arrays
- block: assemble from nested blocks
- trim_zeros: trim leading/trailing zeros
- extract: extract elements where condition is true
- place: place values where condition is true
- putmask: put values using mask
- select: select from choicelist by conditions
- piecewise: piecewise function
- ediff1d: differences with prepend/append
- unwrap: unwrap phase angles
- angle: phase angle of complex numbers
- real_if_close: convert to real if imaginary is small
"""

import numpy as np
import pytest
import rumpy as rp

from conftest import CORE_DTYPES, NUMERIC_DTYPES
from helpers import assert_eq


class TestResize:
    """Test np.resize - resize array with repetition."""

    def test_basic_expand(self):
        """Resize to larger size wraps around."""
        n = np.array([1, 2, 3, 4])
        r = rp.asarray(n)
        assert_eq(rp.resize(r, (6,)), np.resize(n, (6,)))

    def test_basic_shrink(self):
        """Resize to smaller size truncates."""
        n = np.array([1, 2, 3, 4, 5, 6])
        r = rp.asarray(n)
        assert_eq(rp.resize(r, (3,)), np.resize(n, (3,)))

    def test_reshape_expand(self):
        """Resize to different shape."""
        n = np.array([1, 2, 3, 4])
        r = rp.asarray(n)
        assert_eq(rp.resize(r, (2, 3)), np.resize(n, (2, 3)))

    def test_2d_expand(self):
        """Resize 2D array."""
        n = np.array([[1, 2], [3, 4]])
        r = rp.asarray(n)
        assert_eq(rp.resize(r, (3, 3)), np.resize(n, (3, 3)))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, dtype):
        """Test resize preserves dtype."""
        n = np.array([1, 2, 3], dtype=dtype)
        r = rp.asarray(n)
        result = rp.resize(r, (5,))
        expected = np.resize(n, (5,))
        assert_eq(result, expected)

    def test_empty_input(self):
        """Empty input returns empty output."""
        n = np.array([])
        r = rp.asarray(n)
        assert_eq(rp.resize(r, (5,)), np.resize(n, (5,)))


class TestUnstack:
    """Test np.unstack - split along axis into tuple of arrays."""

    def test_basic_axis0(self):
        """Unstack along axis 0."""
        n = np.array([[1, 2], [3, 4], [5, 6]])
        r = rp.asarray(n)
        n_result = np.unstack(n, axis=0)
        r_result = rp.unstack(r, axis=0)
        assert len(r_result) == len(n_result)
        for r_arr, n_arr in zip(r_result, n_result):
            assert_eq(r_arr, n_arr)

    def test_basic_axis1(self):
        """Unstack along axis 1."""
        n = np.array([[1, 2, 3], [4, 5, 6]])
        r = rp.asarray(n)
        n_result = np.unstack(n, axis=1)
        r_result = rp.unstack(r, axis=1)
        assert len(r_result) == len(n_result)
        for r_arr, n_arr in zip(r_result, n_result):
            assert_eq(r_arr, n_arr)

    def test_3d(self):
        """Unstack 3D array."""
        n = np.arange(24).reshape(2, 3, 4)
        r = rp.asarray(n)
        n_result = np.unstack(n, axis=1)
        r_result = rp.unstack(r, axis=1)
        assert len(r_result) == len(n_result)
        for r_arr, n_arr in zip(r_result, n_result):
            assert_eq(r_arr, n_arr)

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_dtypes(self, dtype):
        """Test unstack with different dtypes."""
        n = np.array([[1, 2], [3, 4]], dtype=dtype)
        r = rp.asarray(n)
        n_result = np.unstack(n)
        r_result = rp.unstack(r)
        for r_arr, n_arr in zip(r_result, n_result):
            assert_eq(r_arr, n_arr)


class TestBlock:
    """Test np.block - assemble from nested blocks."""

    def test_1d_simple(self):
        """1D block concatenation."""
        n1, n2 = np.array([1, 2]), np.array([3, 4, 5])
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.block([r1, r2]), np.block([n1, n2]))

    def test_2d_vertical(self):
        """Vertical stacking with block."""
        n1 = np.array([[1, 2]])
        n2 = np.array([[3, 4]])
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.block([[r1], [r2]]), np.block([[n1], [n2]]))

    def test_2d_horizontal(self):
        """Horizontal stacking with block."""
        n1 = np.array([[1], [2]])
        n2 = np.array([[3], [4]])
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.block([[r1, r2]]), np.block([[n1, n2]]))

    def test_2x2_grid(self):
        """2x2 grid block assembly."""
        n1 = np.array([[1, 2], [3, 4]])
        n2 = np.array([[5, 6], [7, 8]])
        n3 = np.array([[9, 10], [11, 12]])
        n4 = np.array([[13, 14], [15, 16]])
        r1, r2, r3, r4 = [rp.asarray(x) for x in [n1, n2, n3, n4]]
        assert_eq(
            rp.block([[r1, r2], [r3, r4]]),
            np.block([[n1, n2], [n3, n4]])
        )

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_dtypes(self, dtype):
        """Test block with different dtypes."""
        n1 = np.array([[1, 2]], dtype=dtype)
        n2 = np.array([[3, 4]], dtype=dtype)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.block([[r1], [r2]]), np.block([[n1], [n2]]))


class TestTrimZeros:
    """Test np.trim_zeros - trim leading/trailing zeros."""

    def test_basic(self):
        """Trim zeros from both ends."""
        n = np.array([0, 0, 1, 2, 3, 0, 0])
        r = rp.asarray(n)
        assert_eq(rp.trim_zeros(r), np.trim_zeros(n))

    def test_trim_front(self):
        """Trim zeros from front only."""
        n = np.array([0, 0, 1, 2, 3, 0, 0])
        r = rp.asarray(n)
        assert_eq(rp.trim_zeros(r, "f"), np.trim_zeros(n, "f"))

    def test_trim_back(self):
        """Trim zeros from back only."""
        n = np.array([0, 0, 1, 2, 3, 0, 0])
        r = rp.asarray(n)
        assert_eq(rp.trim_zeros(r, "b"), np.trim_zeros(n, "b"))

    def test_all_zeros(self):
        """Array of all zeros."""
        n = np.array([0, 0, 0])
        r = rp.asarray(n)
        assert_eq(rp.trim_zeros(r), np.trim_zeros(n))

    def test_no_zeros(self):
        """Array with no zeros."""
        n = np.array([1, 2, 3])
        r = rp.asarray(n)
        assert_eq(rp.trim_zeros(r), np.trim_zeros(n))

    def test_float(self):
        """Float array with zeros."""
        n = np.array([0.0, 0.0, 1.5, 2.5, 0.0])
        r = rp.asarray(n)
        assert_eq(rp.trim_zeros(r), np.trim_zeros(n))


class TestExtract:
    """Test np.extract - extract elements where condition is true."""

    def test_basic(self):
        """Extract elements matching condition."""
        n = np.array([1, 2, 3, 4, 5])
        r = rp.asarray(n)
        cond_n = n > 2
        cond_r = r > 2
        assert_eq(rp.extract(cond_r, r), np.extract(cond_n, n))

    def test_2d(self):
        """Extract from 2D array (flattens result)."""
        n = np.array([[1, 2], [3, 4], [5, 6]])
        r = rp.asarray(n)
        cond_n = n > 3
        cond_r = r > 3
        assert_eq(rp.extract(cond_r, r), np.extract(cond_n, n))

    def test_none_match(self):
        """No elements match condition."""
        n = np.array([1, 2, 3])
        r = rp.asarray(n)
        cond_n = n > 10
        cond_r = r > 10
        assert_eq(rp.extract(cond_r, r), np.extract(cond_n, n))

    def test_all_match(self):
        """All elements match condition."""
        n = np.array([1, 2, 3])
        r = rp.asarray(n)
        cond_n = n > 0
        cond_r = r > 0
        assert_eq(rp.extract(cond_r, r), np.extract(cond_n, n))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, dtype):
        """Test extract with different dtypes."""
        n = np.array([1, 2, 3, 4, 5], dtype=dtype)
        r = rp.asarray(n)
        cond_n = n > 2
        cond_r = r > 2
        assert_eq(rp.extract(cond_r, r), np.extract(cond_n, n))


class TestPlace:
    """Test np.place - place values where condition is true (cycles values)."""

    def test_basic(self):
        """Place values cycling through the list."""
        n = np.array([1, 2, 3, 4, 5]).copy()
        r = rp.array([1, 2, 3, 4, 5])
        np.place(n, n > 2, [10, 20])
        rp.place(r, r > 2, rp.array([10, 20]))
        assert_eq(r, n)

    def test_single_value(self):
        """Place single value."""
        n = np.array([1, 2, 3, 4, 5]).copy()
        r = rp.array([1, 2, 3, 4, 5])
        np.place(n, n > 2, [99])
        rp.place(r, r > 2, rp.array([99]))
        assert_eq(r, n)

    def test_2d(self):
        """Place in 2D array."""
        n = np.array([[1, 2], [3, 4], [5, 6]]).copy()
        r = rp.asarray(n.copy())
        np.place(n, n > 3, [10, 20, 30])
        rp.place(r, r > 3, rp.array([10, 20, 30]))
        assert_eq(r, n)

    def test_none_match(self):
        """No elements match condition - no change."""
        n = np.array([1, 2, 3]).copy()
        r = rp.array([1, 2, 3])
        np.place(n, n > 10, [99])
        rp.place(r, r > 10, rp.array([99]))
        assert_eq(r, n)


class TestPutmask:
    """Test np.putmask - put values using mask (broadcasts values)."""

    def test_basic(self):
        """Putmask broadcasts values."""
        n = np.array([1, 2, 3, 4, 5]).copy()
        r = rp.array([1, 2, 3, 4, 5])
        np.putmask(n, n > 2, [10, 20, 30])
        rp.putmask(r, r > 2, rp.array([10, 20, 30]))
        assert_eq(r, n)

    def test_scalar_value(self):
        """Putmask with scalar."""
        n = np.array([1, 2, 3, 4, 5]).copy()
        r = rp.array([1, 2, 3, 4, 5])
        np.putmask(n, n > 2, 99)
        rp.putmask(r, r > 2, rp.array([99]))
        assert_eq(r, n)

    def test_2d(self):
        """Putmask on 2D array."""
        n = np.array([[1, 2], [3, 4], [5, 6]]).copy()
        r = rp.asarray(n.copy())
        np.putmask(n, n > 3, [10, 20, 30])
        rp.putmask(r, r > 3, rp.array([10, 20, 30]))
        assert_eq(r, n)


class TestSelect:
    """Test np.select - select from choicelist by conditions."""

    def test_basic(self):
        """Select based on conditions."""
        n = np.arange(10)
        r = rp.arange(10)
        condlist_n = [n < 3, n >= 7]
        condlist_r = [r < 3, r >= 7]
        choicelist_n = [n, n ** 2]
        choicelist_r = [r, r ** 2]
        assert_eq(
            rp.select(condlist_r, choicelist_r, default=0),
            np.select(condlist_n, choicelist_n, default=0)
        )

    def test_with_default(self):
        """Select with non-zero default."""
        n = np.array([1, 2, 3, 4, 5])
        r = rp.asarray(n)
        condlist_n = [n < 2, n > 4]
        condlist_r = [r < 2, r > 4]
        choicelist_n = [n * 10, n * 100]
        choicelist_r = [r * 10, r * 100]
        assert_eq(
            rp.select(condlist_r, choicelist_r, default=-1),
            np.select(condlist_n, choicelist_n, default=-1)
        )

    def test_overlapping_conditions(self):
        """First matching condition wins."""
        n = np.array([1, 2, 3, 4, 5])
        r = rp.asarray(n)
        # Both conditions match for 3
        condlist_n = [n >= 3, n >= 2]
        condlist_r = [r >= 3, r >= 2]
        choicelist_n = [n * 10, n * 100]
        choicelist_r = [r * 10, r * 100]
        assert_eq(
            rp.select(condlist_r, choicelist_r),
            np.select(condlist_n, choicelist_n)
        )


class TestPiecewise:
    """Test np.piecewise - evaluate piecewise function."""

    def test_basic_abs(self):
        """Piecewise implementation of abs."""
        n = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        r = rp.asarray(n)
        # abs(x) = -x if x < 0 else x
        condlist_n = [n < 0, n >= 0]
        condlist_r = [r < 0, r >= 0]
        funclist_n = [lambda t: -t, lambda t: t]
        funclist_r = [lambda t: -t, lambda t: t]
        assert_eq(
            rp.piecewise(r, condlist_r, funclist_r),
            np.piecewise(n, condlist_n, funclist_n)
        )

    def test_with_constants(self):
        """Piecewise with constant values instead of functions."""
        n = np.array([-2, -1, 0, 1, 2], dtype=float)
        r = rp.asarray(n)
        condlist_n = [n < 0, n == 0, n > 0]
        condlist_r = [r < 0, r == 0, r > 0]
        funclist = [-1, 0, 1]  # Constants
        assert_eq(
            rp.piecewise(r, condlist_r, funclist),
            np.piecewise(n, condlist_n, funclist)
        )


class TestEdiff1d:
    """Test np.ediff1d - differences with prepend/append."""

    def test_basic(self):
        """Basic differences."""
        n = np.array([1, 3, 6, 10])
        r = rp.asarray(n)
        assert_eq(rp.ediff1d(r), np.ediff1d(n))

    def test_with_prepend(self):
        """Differences with prepend."""
        n = np.array([1, 3, 6, 10])
        r = rp.asarray(n)
        assert_eq(rp.ediff1d(r, to_begin=-1), np.ediff1d(n, to_begin=-1))

    def test_with_append(self):
        """Differences with append."""
        n = np.array([1, 3, 6, 10])
        r = rp.asarray(n)
        assert_eq(rp.ediff1d(r, to_end=100), np.ediff1d(n, to_end=100))

    def test_with_both(self):
        """Differences with both prepend and append."""
        n = np.array([1, 3, 6, 10])
        r = rp.asarray(n)
        assert_eq(
            rp.ediff1d(r, to_begin=-1, to_end=100),
            np.ediff1d(n, to_begin=-1, to_end=100)
        )

    def test_2d_flattens(self):
        """2D array is flattened."""
        n = np.array([[1, 2], [3, 4]])
        r = rp.asarray(n)
        assert_eq(rp.ediff1d(r), np.ediff1d(n))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, dtype):
        """Test ediff1d with different dtypes."""
        n = np.array([1, 3, 6, 10], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.ediff1d(r), np.ediff1d(n))


class TestUnwrap:
    """Test np.unwrap - unwrap phase angles."""

    def test_basic(self):
        """Unwrap phase discontinuities."""
        n = np.array([0, 0.5, 1.0, 1.5, -3.0, -2.5, -2.0])
        r = rp.asarray(n)
        assert_eq(rp.unwrap(r), np.unwrap(n))

    def test_with_discont(self):
        """Unwrap with custom discontinuity threshold."""
        n = np.array([0, 1, 2, -3, -2, -1])
        r = rp.asarray(n)
        assert_eq(rp.unwrap(r, discont=2.0), np.unwrap(n, discont=2.0))

    def test_with_period(self):
        """Unwrap with custom period."""
        n = np.array([0, 1, 2, -3, -2])
        r = rp.asarray(n)
        assert_eq(rp.unwrap(r, period=4.0), np.unwrap(n, period=4.0))

    def test_along_axis(self):
        """Unwrap along specific axis."""
        n = np.array([[0, 0.5, 1.5, -2.5], [0, 1.0, -2.0, -1.0]])
        r = rp.asarray(n)
        assert_eq(rp.unwrap(r, axis=1), np.unwrap(n, axis=1))
        assert_eq(rp.unwrap(r, axis=0), np.unwrap(n, axis=0))

    def test_no_discontinuity(self):
        """Array with no discontinuity."""
        n = np.array([0, 0.1, 0.2, 0.3, 0.4])
        r = rp.asarray(n)
        assert_eq(rp.unwrap(r), np.unwrap(n))


class TestAngle:
    """Test np.angle - phase angle of complex numbers."""

    def test_basic(self):
        """Get phase angles."""
        n = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        r = rp.asarray(n)
        assert_eq(rp.angle(r), np.angle(n))

    def test_degrees(self):
        """Get phase angles in degrees."""
        n = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        r = rp.asarray(n)
        assert_eq(rp.angle(r, deg=True), np.angle(n, deg=True))

    def test_real_array(self):
        """Angle of real numbers."""
        n = np.array([1.0, -1.0, 0.0])
        r = rp.asarray(n)
        assert_eq(rp.angle(r), np.angle(n))

    def test_2d(self):
        """Angle of 2D complex array."""
        n = np.array([[1 + 1j, -1], [1j, -1j]])
        r = rp.asarray(n)
        assert_eq(rp.angle(r), np.angle(n))


class TestRealIfClose:
    """Test np.real_if_close - convert to real if imaginary is small."""

    def test_small_imag(self):
        """Small imaginary parts removed."""
        n = np.array([1 + 1e-10j, 2 + 0j, 3 + 0j])
        r = rp.asarray(n)
        # With default tol, 1e-10 is small enough
        n_result = np.real_if_close(n)
        r_result = rp.real_if_close(r)
        assert_eq(r_result, n_result)

    def test_large_imag(self):
        """Large imaginary parts preserved."""
        n = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        r = rp.asarray(n)
        n_result = np.real_if_close(n)
        r_result = rp.real_if_close(r)
        assert_eq(r_result, n_result)

    def test_with_tol(self):
        """Custom tolerance."""
        n = np.array([1 + 0.01j, 2 + 0.001j])
        r = rp.asarray(n)
        # Large tolerance - should convert
        assert_eq(
            rp.real_if_close(r, tol=100),
            np.real_if_close(n, tol=100)
        )

    def test_real_input(self):
        """Real input passes through."""
        n = np.array([1.0, 2.0, 3.0])
        r = rp.asarray(n)
        assert_eq(rp.real_if_close(r), np.real_if_close(n))
