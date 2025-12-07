"""Tests for polynomial operations (Stream 17)."""

import numpy as np
import pytest
import rumpy as rp
from helpers import assert_eq


class TestPolyval:
    """Tests for polyval - polynomial evaluation."""

    def test_polyval_basic(self):
        """Evaluate polynomial at single points."""
        # x^2 + 2x + 3 at x=0, 1, 2
        coeffs_r = rp.array([1.0, 2.0, 3.0])
        coeffs_n = np.array([1.0, 2.0, 3.0])
        x_r = rp.array([0.0, 1.0, 2.0])
        x_n = np.array([0.0, 1.0, 2.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_polyval_linear(self):
        """Evaluate linear polynomial 2x + 1."""
        coeffs_r = rp.array([2.0, 1.0])
        coeffs_n = np.array([2.0, 1.0])
        x_r = rp.array([-1.0, 0.0, 1.0, 2.0])
        x_n = np.array([-1.0, 0.0, 1.0, 2.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_polyval_constant(self):
        """Evaluate constant polynomial."""
        coeffs_r = rp.array([5.0])
        coeffs_n = np.array([5.0])
        x_r = rp.array([0.0, 1.0, 100.0])
        x_n = np.array([0.0, 1.0, 100.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_polyval_cubic(self):
        """Evaluate cubic polynomial x^3 - 2x^2 + x - 1."""
        coeffs_r = rp.array([1.0, -2.0, 1.0, -1.0])
        coeffs_n = np.array([1.0, -2.0, 1.0, -1.0])
        x_r = rp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        x_n = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_polyval_2d_x(self):
        """Evaluate polynomial at 2D array of points."""
        coeffs_r = rp.array([1.0, 0.0, -1.0])  # x^2 - 1
        coeffs_n = np.array([1.0, 0.0, -1.0])
        x_r = rp.array([[0.0, 1.0], [2.0, 3.0]])
        x_n = np.array([[0.0, 1.0], [2.0, 3.0]])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)


class TestPolyder:
    """Tests for polyder - polynomial derivative."""

    def test_polyder_quadratic(self):
        """Derivative of x^2 + 2x + 3 is 2x + 2."""
        p_r = rp.array([1.0, 2.0, 3.0])
        p_n = np.array([1.0, 2.0, 3.0])

        r = rp.polyder(p_r)
        n = np.polyder(p_n)
        assert_eq(r, n)

    def test_polyder_cubic(self):
        """Derivative of x^3 - 2x^2 + x - 1 is 3x^2 - 4x + 1."""
        p_r = rp.array([1.0, -2.0, 1.0, -1.0])
        p_n = np.array([1.0, -2.0, 1.0, -1.0])

        r = rp.polyder(p_r)
        n = np.polyder(p_n)
        assert_eq(r, n)

    def test_polyder_second_derivative(self):
        """Second derivative of x^3 is 6x."""
        p_r = rp.array([1.0, 0.0, 0.0, 0.0])  # x^3
        p_n = np.array([1.0, 0.0, 0.0, 0.0])

        r = rp.polyder(p_r, m=2)
        n = np.polyder(p_n, 2)
        assert_eq(r, n)

    def test_polyder_constant(self):
        """Derivative of constant is empty array."""
        p_r = rp.array([5.0])
        p_n = np.array([5.0])

        r = rp.polyder(p_r)
        n = np.polyder(p_n)
        assert_eq(r, n)

    def test_polyder_linear(self):
        """Derivative of 2x + 1 is 2."""
        p_r = rp.array([2.0, 1.0])
        p_n = np.array([2.0, 1.0])

        r = rp.polyder(p_r)
        n = np.polyder(p_n)
        assert_eq(r, n)


class TestPolyint:
    """Tests for polyint - polynomial integral."""

    def test_polyint_linear(self):
        """Integral of 2x + 1 is x^2 + x + c."""
        p_r = rp.array([2.0, 1.0])
        p_n = np.array([2.0, 1.0])

        r = rp.polyint(p_r)
        n = np.polyint(p_n)
        assert_eq(r, n)

    def test_polyint_quadratic(self):
        """Integral of x^2 + 2x + 3."""
        p_r = rp.array([1.0, 2.0, 3.0])
        p_n = np.array([1.0, 2.0, 3.0])

        r = rp.polyint(p_r)
        n = np.polyint(p_n)
        assert_eq(r, n)

    def test_polyint_with_constant(self):
        """Integral with integration constant."""
        p_r = rp.array([1.0, 2.0, 3.0])
        p_n = np.array([1.0, 2.0, 3.0])
        k_r = rp.array([5.0])

        r = rp.polyint(p_r, k=k_r)
        n = np.polyint(p_n, k=5.0)
        assert_eq(r, n)

    def test_polyint_multiple_integrals(self):
        """Double integral of x is x^3/6 + c1*x + c0."""
        p_r = rp.array([1.0, 0.0])  # x
        p_n = np.array([1.0, 0.0])

        r = rp.polyint(p_r, m=2)
        n = np.polyint(p_n, m=2)
        assert_eq(r, n)


class TestPolyfit:
    """Tests for polyfit - polynomial fitting."""

    def test_polyfit_linear(self):
        """Fit line to linear data."""
        x_r = rp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x_n = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # y = 2x + 1
        y_r = rp.array([1.0, 3.0, 5.0, 7.0, 9.0])
        y_n = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

        r = rp.polyfit(x_r, y_r, 1)
        n = np.polyfit(x_n, y_n, 1)
        assert_eq(r, n, rtol=1e-5)

    def test_polyfit_quadratic(self):
        """Fit quadratic to quadratic data."""
        x_r = rp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x_n = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # y = x^2 + x + 1
        y_r = rp.array([1.0, 3.0, 7.0, 13.0, 21.0])
        y_n = np.array([1.0, 3.0, 7.0, 13.0, 21.0])

        r = rp.polyfit(x_r, y_r, 2)
        n = np.polyfit(x_n, y_n, 2)
        assert_eq(r, n, rtol=1e-5)

    def test_polyfit_overdetermined(self):
        """Fit lower-degree polynomial to noisy data."""
        x_r = rp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        x_n = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        # Approximately y = 2x + 1 with some noise
        y_r = rp.array([1.1, 2.9, 5.1, 6.9, 9.1, 10.9])
        y_n = np.array([1.1, 2.9, 5.1, 6.9, 9.1, 10.9])

        r = rp.polyfit(x_r, y_r, 1)
        n = np.polyfit(x_n, y_n, 1)
        assert_eq(r, n, rtol=1e-5)


class TestRoots:
    """Tests for roots - polynomial roots."""

    def test_roots_quadratic_real(self):
        """Roots of x^2 - 4 are +2 and -2."""
        # x^2 - 4 = (x-2)(x+2)
        p_r = rp.array([1.0, 0.0, -4.0])
        p_n = np.array([1.0, 0.0, -4.0])

        r = rp.roots(p_r)
        n = np.roots(p_n)

        # Both should be real: +2 and -2
        r_np = np.asarray(r)
        # Sort for comparison (roots may be in different order)
        r_sorted = np.sort(r_np)
        n_sorted = np.sort(n)
        assert_eq(r_sorted, n_sorted, rtol=1e-5)

    def test_roots_cubic_integer(self):
        """Roots of (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6."""
        p_r = rp.array([1.0, -6.0, 11.0, -6.0])
        p_n = np.array([1.0, -6.0, 11.0, -6.0])

        r = rp.roots(p_r)
        n = np.roots(p_n)

        r_np = np.asarray(r)
        r_sorted = np.sort(r_np)
        n_sorted = np.sort(n.real)  # NumPy may return complex with 0 imaginary
        assert_eq(r_sorted, n_sorted, rtol=1e-3)

    def test_roots_linear(self):
        """Root of 2x + 4 is -2."""
        p_r = rp.array([2.0, 4.0])
        p_n = np.array([2.0, 4.0])

        r = rp.roots(p_r)
        n = np.roots(p_n)

        assert_eq(r, n.real, rtol=1e-5)

    def test_roots_constant_no_roots(self):
        """Constant polynomial has no roots."""
        p_r = rp.array([5.0])

        r = rp.roots(p_r)
        r_np = np.asarray(r)
        assert r_np.size == 0


class TestPolyRoundtrip:
    """Tests for polynomial operation round-trips."""

    def test_derivative_integral_roundtrip(self):
        """Integral of derivative recovers original (up to constant)."""
        # Start with p(x) = x^2 + 2x + 3
        p = rp.array([1.0, 2.0, 3.0])

        # d/dx p(x) = 2x + 2
        dp = rp.polyder(p)

        # Integrate: should get x^2 + 2x + c
        # With c=0 by default, won't match original constant term
        integ = rp.polyint(dp)

        # The first two coefficients should match
        p_np = np.asarray(p)
        integ_np = np.asarray(integ)

        # integ = [1, 2, 0] vs p = [1, 2, 3]
        # First two match
        np.testing.assert_allclose(integ_np[:-1], p_np[:-1], rtol=1e-10)

    def test_fit_eval_roundtrip(self):
        """Fitting and evaluating should reproduce original values."""
        x_r = rp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_r = rp.array([1.0, 3.0, 7.0, 13.0, 21.0])  # x^2 + x + 1

        # Fit degree 2
        coeffs = rp.polyfit(x_r, y_r, 2)

        # Evaluate at same points
        y_pred = rp.polyval(coeffs, x_r)

        # Should match original y
        assert_eq(y_pred, y_r, rtol=1e-5)
