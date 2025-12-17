"""Tests for functional programming operations: apply_along_axis, apply_over_axes, vectorize, frompyfunc."""

import numpy as np
import pytest

import rumpy as rp

from helpers import assert_eq


def np_sum(x):
    """Wrapper for numpy sum that converts rumpy arrays."""
    return np.sum(np.asarray(x))

class TestApplyAlongAxis:
    """Tests for apply_along_axis."""

    def test_sum_axis0(self):
        """Apply sum along axis 0."""
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rp.asarray(n)
        assert_eq(
            rp.apply_along_axis(np_sum, 0, r),
            np.apply_along_axis(np_sum, 0, n),
        )

    def test_sum_axis1(self):
        """Apply sum along axis 1."""
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rp.asarray(n)
        assert_eq(
            rp.apply_along_axis(np_sum, 1, r),
            np.apply_along_axis(np_sum, 1, n),
        )

    def test_custom_function(self):
        """Apply custom function returning scalar."""

        def first_plus_last(x):
            arr = np.asarray(x)
            return arr[0] + arr[-1]

        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rp.asarray(n)
        assert_eq(
            rp.apply_along_axis(first_plus_last, 1, r),
            np.apply_along_axis(first_plus_last, 1, n),
        )

    def test_3d_array(self):
        """Apply function along axis in 3D array."""
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)

        assert_eq(
            rp.apply_along_axis(np_sum, 0, r),
            np.apply_along_axis(np_sum, 0, n),
        )
        assert_eq(
            rp.apply_along_axis(np_sum, 1, r),
            np.apply_along_axis(np_sum, 1, n),
        )
        assert_eq(
            rp.apply_along_axis(np_sum, 2, r),
            np.apply_along_axis(np_sum, 2, n),
        )

    def test_negative_axis(self):
        """Apply function along negative axis."""
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rp.asarray(n)
        assert_eq(
            rp.apply_along_axis(np_sum, -1, r),
            np.apply_along_axis(np_sum, -1, n),
        )

    def test_function_returning_array(self):
        """Function that returns array (changes shape)."""

        def double_elements(x):
            return np.concatenate([x, x])

        n = np.array([[1, 2], [3, 4]]).astype(float)
        r = rp.asarray(n)
        assert_eq(
            rp.apply_along_axis(double_elements, 1, r),
            np.apply_along_axis(double_elements, 1, n),
        )

    def test_with_kwargs(self):
        """Pass keyword arguments to function."""

        def power_sum(x, power=1):
            arr = np.asarray(x)
            return sum(xi**power for xi in arr)

        n = np.array([[1, 2, 3], [4, 5, 6]]).astype(float)
        r = rp.asarray(n)
        # numpy apply_along_axis doesn't support kwargs directly,
        # but we can test passing args via lambda
        assert_eq(
            rp.apply_along_axis(lambda x: power_sum(x, power=2), 1, r),
            np.apply_along_axis(lambda x: power_sum(x, power=2), 1, n),
        )

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
    def test_dtypes(self, dtype):
        """Test with different dtypes."""
        n = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        r = rp.asarray(n)
        # sum may return different dtype, convert for comparison
        result_r = np.asarray(rp.apply_along_axis(np_sum, 1, r))
        result_n = np.apply_along_axis(np_sum, 1, n)
        np.testing.assert_array_equal(result_r, result_n)


def np_sum_axis(x, axis):
    """Wrapper for numpy sum that converts rumpy arrays."""
    return np.sum(np.asarray(x), axis=axis, keepdims=True)


def np_mean_axis(x, axis):
    """Wrapper for numpy mean that converts rumpy arrays."""
    return np.mean(np.asarray(x), axis=axis, keepdims=True)


class TestApplyOverAxes:
    """Tests for apply_over_axes."""

    def test_sum_single_axis(self):
        """Apply sum over single axis."""
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)
        assert_eq(
            rp.apply_over_axes(np_sum_axis, r, [0]),
            np.apply_over_axes(np.sum, n, [0]),
        )

    def test_sum_multiple_axes(self):
        """Apply sum over multiple axes."""
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)
        assert_eq(
            rp.apply_over_axes(np_sum_axis, r, [0, 2]),
            np.apply_over_axes(np.sum, n, [0, 2]),
        )

    def test_mean_over_axes(self):
        """Apply mean over axes."""
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)
        assert_eq(
            rp.apply_over_axes(np_mean_axis, r, [0, 1]),
            np.apply_over_axes(np.mean, n, [0, 1]),
        )

    def test_2d_array(self):
        """Apply over axes on 2D array."""
        n = np.array([[1, 2, 3], [4, 5, 6]]).astype(float)
        r = rp.asarray(n)
        assert_eq(
            rp.apply_over_axes(np_sum_axis, r, [0]),
            np.apply_over_axes(np.sum, n, [0]),
        )
        assert_eq(
            rp.apply_over_axes(np_sum_axis, r, [1]),
            np.apply_over_axes(np.sum, n, [1]),
        )

    def test_negative_axes(self):
        """Apply over negative axes."""
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)
        assert_eq(
            rp.apply_over_axes(np_sum_axis, r, [-1]),
            np.apply_over_axes(np.sum, n, [-1]),
        )

    def test_all_axes(self):
        """Apply over all axes."""
        n = np.arange(24).reshape(2, 3, 4).astype(float)
        r = rp.asarray(n)
        assert_eq(
            rp.apply_over_axes(np_sum_axis, r, [0, 1, 2]),
            np.apply_over_axes(np.sum, n, [0, 1, 2]),
        )


class TestVectorize:
    """Tests for vectorize."""

    def test_simple_function(self):
        """Vectorize simple scalar function."""

        def add_one(x):
            return x + 1

        vfunc_np = np.vectorize(add_one)
        vfunc_rp = rp.vectorize(add_one)

        n = np.array([1, 2, 3, 4])
        assert_eq(vfunc_rp(n), vfunc_np(n))

    def test_conditional_function(self):
        """Vectorize function with conditionals."""

        def myfunc(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        vfunc_np = np.vectorize(myfunc)
        vfunc_rp = rp.vectorize(myfunc)

        a = np.array([1, 2, 3, 4])
        b = np.array([4, 3, 2, 1])
        assert_eq(vfunc_rp(a, b), vfunc_np(a, b))

    def test_with_otypes(self):
        """Vectorize with explicit output types."""

        def square(x):
            return x**2

        vfunc_np = np.vectorize(square, otypes=["float64"])
        vfunc_rp = rp.vectorize(square, otypes=["float64"])

        n = np.array([1, 2, 3])
        result_rp = vfunc_rp(n)
        result_np = vfunc_np(n)
        assert_eq(result_rp, result_np)
        assert result_rp.dtype == result_np.dtype

    def test_broadcasting(self):
        """Vectorize handles broadcasting."""

        def multiply(a, b):
            return a * b

        vfunc_np = np.vectorize(multiply)
        vfunc_rp = rp.vectorize(multiply)

        a = np.array([[1], [2], [3]])
        b = np.array([1, 2, 3])
        assert_eq(vfunc_rp(a, b), vfunc_np(a, b))

    def test_with_excluded(self):
        """Vectorize with excluded arguments."""

        def power(x, n):
            return x**n

        vfunc_np = np.vectorize(power, excluded=["n"])
        vfunc_rp = rp.vectorize(power, excluded=["n"])

        x = np.array([1, 2, 3, 4])
        assert_eq(vfunc_rp(x, n=2), vfunc_np(x, n=2))

    def test_with_signature(self):
        """Vectorize with signature for array inputs."""

        def inner_prod(a, b):
            a_np = np.asarray(a)
            b_np = np.asarray(b)
            return np.sum(a_np * b_np)

        vfunc_np = np.vectorize(inner_prod, signature="(n),(n)->()")
        vfunc_rp = rp.vectorize(inner_prod, signature="(n),(n)->()")

        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[1, 1, 1], [2, 2, 2]])
        assert_eq(vfunc_rp(a, b), vfunc_np(a, b))

    def test_scalar_input(self):
        """Vectorize works with scalar input."""

        def double(x):
            return x * 2

        vfunc_np = np.vectorize(double)
        vfunc_rp = rp.vectorize(double)

        assert vfunc_rp(5) == vfunc_np(5)


class TestFrompyfunc:
    """Tests for frompyfunc."""

    def test_binary_function(self):
        """Create ufunc from binary function."""

        def my_add(a, b):
            return a + b

        ufunc_np = np.frompyfunc(my_add, 2, 1)
        ufunc_rp = rp.frompyfunc(my_add, 2, 1)

        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        # frompyfunc returns object dtype in numpy
        result_rp = np.asarray(ufunc_rp(a, b))
        result_np = ufunc_np(a, b)
        np.testing.assert_array_equal(result_rp.astype(int), result_np.astype(int))

    def test_unary_function(self):
        """Create ufunc from unary function."""

        def square(x):
            return x * x

        ufunc_np = np.frompyfunc(square, 1, 1)
        ufunc_rp = rp.frompyfunc(square, 1, 1)

        a = np.array([1, 2, 3, 4])

        result_rp = np.asarray(ufunc_rp(a))
        result_np = ufunc_np(a)
        np.testing.assert_array_equal(result_rp.astype(int), result_np.astype(int))

    def test_scalar_input(self):
        """frompyfunc with scalar input."""

        def double(x):
            return x * 2

        ufunc_np = np.frompyfunc(double, 1, 1)
        ufunc_rp = rp.frompyfunc(double, 1, 1)

        assert ufunc_rp(5) == ufunc_np(5)

    def test_multiple_outputs(self):
        """Create ufunc with multiple outputs."""

        def divmod_func(a, b):
            return a // b, a % b

        ufunc_np = np.frompyfunc(divmod_func, 2, 2)
        ufunc_rp = rp.frompyfunc(divmod_func, 2, 2)

        a = np.array([10, 20, 30])
        b = np.array([3, 7, 4])

        q_rp, r_rp = ufunc_rp(a, b)
        q_np, r_np = ufunc_np(a, b)

        np.testing.assert_array_equal(
            np.asarray(q_rp).astype(int), q_np.astype(int)
        )
        np.testing.assert_array_equal(
            np.asarray(r_rp).astype(int), r_np.astype(int)
        )

    def test_broadcasting(self):
        """frompyfunc handles broadcasting."""

        def multiply(a, b):
            return a * b

        ufunc_np = np.frompyfunc(multiply, 2, 1)
        ufunc_rp = rp.frompyfunc(multiply, 2, 1)

        a = np.array([[1], [2], [3]])
        b = np.array([1, 2, 3])

        result_rp = np.asarray(ufunc_rp(a, b))
        result_np = ufunc_np(a, b)
        np.testing.assert_array_equal(result_rp.astype(int), result_np.astype(int))
