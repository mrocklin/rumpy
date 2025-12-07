"""Tests for Stream 16: Numerical Operations - gradient, trapezoid, interp, correlate."""

import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestGradient:
    """Test numerical gradient computation."""

    def test_1d_simple(self):
        """Test 1D gradient with uniform spacing."""
        r = rp.gradient(rp.array([1.0, 2.0, 4.0, 7.0, 11.0]))
        n = np.gradient([1, 2, 4, 7, 11])
        assert_eq(r, n)

    def test_1d_with_spacing(self):
        """Test 1D gradient with explicit spacing."""
        r = rp.gradient(rp.array([1.0, 2.0, 4.0, 7.0, 11.0]), 2.0)
        n = np.gradient([1, 2, 4, 7, 11], 2)
        assert_eq(r, n)

    def test_1d_with_coord_array(self):
        """Test 1D gradient with coordinate array."""
        x = [0.0, 1.0, 3.0, 6.0, 10.0]  # Non-uniform spacing
        f = [1.0, 2.0, 4.0, 7.0, 11.0]
        r = rp.gradient(rp.array(f), rp.array(x))
        n = np.gradient(f, x)
        assert_eq(r, n)

    def test_2d_default(self):
        """Test 2D gradient (returns gradient along each axis)."""
        arr = rp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        n_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        r0, r1 = rp.gradient(arr)
        n0, n1 = np.gradient(n_arr)
        assert_eq(r0, n0)
        assert_eq(r1, n1)

    def test_2d_along_axis(self):
        """Test 2D gradient along specific axis."""
        arr = rp.array([[1.0, 2.0, 4.0], [4.0, 6.0, 8.0]])
        n_arr = np.array([[1, 2, 4], [4, 6, 8]], dtype=float)
        r = rp.gradient(arr, axis=1)
        n = np.gradient(n_arr, axis=1)
        assert_eq(r, n)

    def test_single_element(self):
        """Gradient of single element should raise error like numpy."""
        import pytest
        with pytest.raises(ValueError):
            rp.gradient(rp.array([5.0]))

    def test_two_elements(self):
        """Gradient of two elements uses forward/backward difference."""
        r = rp.gradient(rp.array([1.0, 3.0]))
        n = np.gradient([1, 3])
        assert_eq(r, n)


class TestTrapezoid:
    """Test trapezoidal integration."""

    def test_simple(self):
        """Simple integration of y values."""
        y = [1.0, 2.0, 3.0, 4.0]
        r = rp.trapezoid(rp.array(y))
        n = np.trapezoid(y)
        assert abs(float(r) - n) < 1e-10

    def test_with_x(self):
        """Integration with explicit x coordinates."""
        y = [1.0, 2.0, 3.0]
        x = [0.0, 1.0, 3.0]  # Non-uniform
        r = rp.trapezoid(rp.array(y), rp.array(x))
        n = np.trapezoid(y, x)
        assert abs(float(r) - n) < 1e-10

    def test_with_dx(self):
        """Integration with explicit spacing."""
        y = [1.0, 2.0, 3.0, 4.0]
        r = rp.trapezoid(rp.array(y), dx=0.5)
        n = np.trapezoid(y, dx=0.5)
        assert abs(float(r) - n) < 1e-10

    def test_2d_default_axis(self):
        """2D integration along default axis (-1)."""
        y = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        r = rp.trapezoid(rp.array(y))
        n = np.trapezoid(y)
        assert_eq(r, n)

    def test_2d_axis0(self):
        """2D integration along axis 0."""
        y = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        r = rp.trapezoid(rp.array(y), axis=0)
        n = np.trapezoid(y, axis=0)
        assert_eq(r, n)

    def test_empty(self):
        """Empty array gives 0."""
        r = rp.trapezoid(rp.array([]))
        assert float(r) == 0.0

    def test_single(self):
        """Single element gives 0."""
        r = rp.trapezoid(rp.array([5.0]))
        assert float(r) == 0.0


class TestInterp:
    """Test 1D linear interpolation."""

    def test_simple(self):
        """Basic interpolation."""
        xp = [0.0, 1.0, 2.0]
        fp = [0.0, 1.0, 4.0]
        x = [0.5, 1.5]
        r = rp.interp(rp.array(x), rp.array(xp), rp.array(fp))
        n = np.interp(x, xp, fp)
        assert_eq(r, n)

    def test_extrapolation_left(self):
        """Extrapolation to left returns first value."""
        xp = [1.0, 2.0, 3.0]
        fp = [10.0, 20.0, 30.0]
        x = [0.0]
        r = rp.interp(rp.array(x), rp.array(xp), rp.array(fp))
        n = np.interp(x, xp, fp)
        assert_eq(r, n)

    def test_extrapolation_right(self):
        """Extrapolation to right returns last value."""
        xp = [1.0, 2.0, 3.0]
        fp = [10.0, 20.0, 30.0]
        x = [5.0]
        r = rp.interp(rp.array(x), rp.array(xp), rp.array(fp))
        n = np.interp(x, xp, fp)
        assert_eq(r, n)

    def test_exact_match(self):
        """Interpolation at exact data points."""
        xp = [0.0, 1.0, 2.0]
        fp = [0.0, 10.0, 20.0]
        x = [0.0, 1.0, 2.0]
        r = rp.interp(rp.array(x), rp.array(xp), rp.array(fp))
        n = np.interp(x, xp, fp)
        assert_eq(r, n)

    def test_with_left_right(self):
        """Custom left/right extrapolation values."""
        xp = [1.0, 2.0, 3.0]
        fp = [10.0, 20.0, 30.0]
        x = [0.0, 4.0]
        r = rp.interp(rp.array(x), rp.array(xp), rp.array(fp), left=-1.0, right=-2.0)
        n = np.interp(x, xp, fp, left=-1, right=-2)
        assert_eq(r, n)

    def test_scalar_x(self):
        """Single x value."""
        xp = [0.0, 1.0, 2.0]
        fp = [0.0, 10.0, 20.0]
        r = rp.interp(rp.array([0.5]), rp.array(xp), rp.array(fp))
        n = np.interp([0.5], xp, fp)
        assert_eq(r, n)


class TestCorrelate:
    """Test cross-correlation."""

    def test_full_mode(self):
        """Full cross-correlation."""
        a = [1.0, 2.0, 3.0]
        v = [0.5, 0.5]
        r = rp.correlate(rp.array(a), rp.array(v), 'full')
        n = np.correlate(a, v, 'full')
        assert_eq(r, n)

    def test_same_mode(self):
        """Same-size cross-correlation."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        v = [0.25, 0.5, 0.25]
        r = rp.correlate(rp.array(a), rp.array(v), 'same')
        n = np.correlate(a, v, 'same')
        assert_eq(r, n)

    def test_valid_mode(self):
        """Valid cross-correlation (no padding)."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        v = [1.0, 2.0, 1.0]
        r = rp.correlate(rp.array(a), rp.array(v), 'valid')
        n = np.correlate(a, v, 'valid')
        assert_eq(r, n)

    def test_autocorrelation(self):
        """Autocorrelation (array with itself)."""
        a = [1.0, 2.0, 3.0]
        r = rp.correlate(rp.array(a), rp.array(a), 'full')
        n = np.correlate(a, a, 'full')
        assert_eq(r, n)

    def test_different_from_convolve(self):
        """Correlate differs from convolve (v is not reversed)."""
        a = [1.0, 2.0, 3.0]
        v = [1.0, 2.0]
        r_corr = rp.correlate(rp.array(a), rp.array(v), 'full')
        r_conv = rp.convolve(rp.array(a), rp.array(v), 'full')
        n_corr = np.correlate(a, v, 'full')
        n_conv = np.convolve(a, v, 'full')
        assert_eq(r_corr, n_corr)
        assert_eq(r_conv, n_conv)
        # They should be different when v is not symmetric
        assert not np.allclose(np.asarray(r_corr), np.asarray(r_conv))
