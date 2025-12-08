"""Tests for bitwise operations.

Parametrizes over operations for concise comprehensive coverage.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import BITWISE_DTYPES, CORE_SHAPES, INT_DTYPES, SHAPES_BROADCAST
from helpers import assert_eq, make_pair

# === Bitwise operation categories ===

# Binary bitwise ops
BINARY_BITWISE_OPS = ["bitwise_and", "bitwise_or", "bitwise_xor"]

# Unary bitwise ops
UNARY_BITWISE_OPS = ["bitwise_not", "invert"]

# Shift ops
SHIFT_OPS = ["left_shift", "right_shift"]


# === Parametrized tests ===


class TestBinaryBitwiseOps:
    """Test binary bitwise operations: and, or, xor."""

    @pytest.mark.parametrize("op", BINARY_BITWISE_OPS)
    @pytest.mark.parametrize("dtype", BITWISE_DTYPES)
    def test_dtypes(self, op, dtype):
        """Test all bitwise dtypes with basic patterns."""
        a = np.array([0b1100, 0b1010, 0b1111, 0b0000], dtype=dtype)
        b = np.array([0b1010, 0b1100, 0b0000, 0b1111], dtype=dtype)
        ra, rb = rp.asarray(a), rp.asarray(b)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra, rb), np_fn(a, b))

    @pytest.mark.parametrize("op", BINARY_BITWISE_OPS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, op, shape):
        """Test various shapes."""
        ra, a = make_pair(shape, "int64")
        rb, b = make_pair(shape, "int64")
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra, rb), np_fn(a, b))

    @pytest.mark.parametrize("op", BINARY_BITWISE_OPS)
    def test_scalar(self, op):
        """Test scalar broadcasting."""
        a = np.array([0b1100, 0b1010, 0b1111], dtype=np.int64)
        scalar = np.int64(0b1010)
        ra = rp.asarray(a)
        r_scalar = rp.asarray(scalar)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra, r_scalar), np_fn(a, scalar))
        assert_eq(rp_fn(r_scalar, ra), np_fn(scalar, a))

    @pytest.mark.parametrize("op", BINARY_BITWISE_OPS)
    @pytest.mark.parametrize("shapes", SHAPES_BROADCAST)
    def test_broadcast(self, op, shapes):
        """Test broadcasting of binary bitwise ops."""
        shape1, shape2 = shapes
        a = np.ones(shape1, dtype=np.int64) * 0b1100
        b = np.ones(shape2, dtype=np.int64) * 0b1010
        ra, rb = rp.asarray(a), rp.asarray(b)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra, rb), np_fn(a, b))

    @pytest.mark.parametrize("op", BINARY_BITWISE_OPS)
    def test_bool_dtype(self, op):
        """Test boolean arrays (special case for bitwise ops)."""
        a = np.array([True, False, True, False], dtype=bool)
        b = np.array([True, True, False, False], dtype=bool)
        ra, rb = rp.asarray(a), rp.asarray(b)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra, rb), np_fn(a, b))


class TestUnaryBitwiseOps:
    """Test unary bitwise operations: not, invert."""

    @pytest.mark.parametrize("op", UNARY_BITWISE_OPS)
    @pytest.mark.parametrize("dtype", BITWISE_DTYPES)
    def test_dtypes(self, op, dtype):
        """Test all bitwise dtypes."""
        a = np.array([0b1100, 0b1010, 0b1111, 0b0000], dtype=dtype)
        ra = rp.asarray(a)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra), np_fn(a))

    @pytest.mark.parametrize("op", UNARY_BITWISE_OPS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, op, shape):
        """Test various shapes."""
        ra, a = make_pair(shape, "int64")
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra), np_fn(a))

    @pytest.mark.parametrize("op", UNARY_BITWISE_OPS)
    def test_bool_dtype(self, op):
        """Test boolean arrays."""
        a = np.array([True, False, True, False], dtype=bool)
        ra = rp.asarray(a)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra), np_fn(a))


class TestShiftOps:
    """Test shift operations: left_shift, right_shift."""

    @pytest.mark.parametrize("op", SHIFT_OPS)
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_dtypes(self, op, dtype):
        """Test signed integer dtypes."""
        a = np.array([1, 2, 4, 8], dtype=dtype)
        shifts = np.array([1, 2, 1, 3], dtype=dtype)
        ra, rshifts = rp.asarray(a), rp.asarray(shifts)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra, rshifts), np_fn(a, shifts))

    @pytest.mark.parametrize("op", SHIFT_OPS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, op, shape):
        """Test various shapes."""
        # Use positive values for shifts
        size = int(np.prod(shape))
        a = np.arange(1, size + 1, dtype=np.int64).reshape(shape)
        shifts = np.ones(shape, dtype=np.int64) * 2
        ra, rshifts = rp.asarray(a), rp.asarray(shifts)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra, rshifts), np_fn(a, shifts))

    @pytest.mark.parametrize("op", SHIFT_OPS)
    def test_scalar_shift(self, op):
        """Test shifting by scalar."""
        a = np.array([1, 2, 4, 8, 16], dtype=np.int64)
        shift = np.int64(2)
        ra = rp.asarray(a)
        r_shift = rp.asarray(shift)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(ra, r_shift), np_fn(a, shift))

    def test_left_shift_amounts(self):
        """Test various left shift amounts."""
        a = np.array([1, 1, 1, 1], dtype=np.int64)
        ra = rp.asarray(a)
        for shift in [0, 1, 4, 8, 16]:
            r_shift = rp.asarray(np.int64(shift))
            assert_eq(rp.left_shift(ra, r_shift), np.left_shift(a, shift))

    def test_right_shift_amounts(self):
        """Test various right shift amounts."""
        a = np.array([256, 256, 256, 256], dtype=np.int64)
        ra = rp.asarray(a)
        for shift in [0, 1, 2, 4, 8]:
            r_shift = rp.asarray(np.int64(shift))
            assert_eq(rp.right_shift(ra, r_shift), np.right_shift(a, shift))

    def test_right_shift_negative(self):
        """Test right shift preserves sign (arithmetic shift)."""
        a = np.array([-8, -16, -32], dtype=np.int64)
        shift = np.int64(2)
        ra = rp.asarray(a)
        r_shift = rp.asarray(shift)
        assert_eq(rp.right_shift(ra, r_shift), np.right_shift(a, shift))


# === Operator overloading (dunder methods) ===


class TestBitwiseDunderOps:
    """Test bitwise operator overloading matches numpy."""

    def test_and_op(self):
        """Test & operator."""
        a = np.array([0b1100, 0b1010, 0b1111], dtype=np.int64)
        b = np.array([0b1010, 0b1100, 0b0000], dtype=np.int64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(ra & rb, a & b)

    def test_or_op(self):
        """Test | operator."""
        a = np.array([0b1100, 0b1010, 0b0000], dtype=np.int64)
        b = np.array([0b1010, 0b0100, 0b1111], dtype=np.int64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(ra | rb, a | b)

    def test_xor_op(self):
        """Test ^ operator."""
        a = np.array([0b1100, 0b1010, 0b1111], dtype=np.int64)
        b = np.array([0b1010, 0b1010, 0b0000], dtype=np.int64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(ra ^ rb, a ^ b)

    def test_invert_op(self):
        """Test ~ operator."""
        a = np.array([0, 1, -1, 5], dtype=np.int64)
        ra = rp.asarray(a)
        assert_eq(~ra, ~a)

    def test_lshift_op(self):
        """Test << operator."""
        a = np.array([1, 2, 4, 8], dtype=np.int64)
        ra = rp.asarray(a)
        assert_eq(ra << 2, a << 2)

    def test_rshift_op(self):
        """Test >> operator."""
        a = np.array([8, 16, 32, 64], dtype=np.int64)
        ra = rp.asarray(a)
        assert_eq(ra >> 2, a >> 2)

    def test_and_bool(self):
        """Test & with boolean arrays."""
        a = np.array([True, False, True, False], dtype=bool)
        b = np.array([True, True, False, False], dtype=bool)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(ra & rb, a & b)

    def test_or_bool(self):
        """Test | with boolean arrays."""
        a = np.array([True, False, True, False], dtype=bool)
        b = np.array([True, True, False, False], dtype=bool)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(ra | rb, a | b)

    def test_xor_bool(self):
        """Test ^ with boolean arrays."""
        a = np.array([True, False, True, False], dtype=bool)
        b = np.array([True, True, False, False], dtype=bool)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(ra ^ rb, a ^ b)

    def test_invert_bool(self):
        """Test ~ with boolean arrays."""
        a = np.array([True, False, True, False], dtype=bool)
        ra = rp.asarray(a)
        assert_eq(~ra, ~a)

    def test_and_scalar(self):
        """Test & with scalar."""
        a = np.array([0b1111, 0b1010, 0b0101], dtype=np.int64)
        ra = rp.asarray(a)
        assert_eq(ra & 0b1100, a & 0b1100)

    def test_lshift_scalar(self):
        """Test << with scalar."""
        a = np.array([1, 2, 3], dtype=np.int64)
        ra = rp.asarray(a)
        assert_eq(ra << 2, a << 2)

    def test_rshift_scalar(self):
        """Test >> with scalar."""
        a = np.array([8, 16, 32], dtype=np.int64)
        ra = rp.asarray(a)
        assert_eq(ra >> 2, a >> 2)

    def test_rlshift_scalar_array(self):
        """Test scalar << array."""
        a = np.array([1, 2, 3], dtype=np.int64)
        ra = rp.asarray(a)
        assert_eq(1 << ra, 1 << a)

    def test_rrshift_scalar_array(self):
        """Test scalar >> array."""
        a = np.array([1, 2, 3], dtype=np.int64)
        ra = rp.asarray(a)
        assert_eq(64 >> ra, 64 >> a)


# === Edge cases ===


class TestBitwiseEdgeCases:
    """Test edge cases for bitwise operations."""

    def test_empty_array(self):
        """Test bitwise ops on empty arrays."""
        a = np.array([], dtype=np.int64)
        b = np.array([], dtype=np.int64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.bitwise_and(ra, rb), np.bitwise_and(a, b))
        assert_eq(rp.bitwise_or(ra, rb), np.bitwise_or(a, b))
        assert_eq(rp.bitwise_xor(ra, rb), np.bitwise_xor(a, b))
        assert_eq(rp.bitwise_not(ra), np.bitwise_not(a))

    def test_all_zeros(self):
        """Test with all zeros."""
        a = np.zeros(5, dtype=np.int64)
        b = np.array([0b1010, 0b1100, 0b1111, 0b0001, 0b0101], dtype=np.int64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.bitwise_and(ra, rb), np.bitwise_and(a, b))
        assert_eq(rp.bitwise_or(ra, rb), np.bitwise_or(a, b))
        assert_eq(rp.bitwise_xor(ra, rb), np.bitwise_xor(a, b))

    def test_all_ones_pattern(self):
        """Test with all bits set."""
        a = np.array([0xFF, 0xFF, 0xFF], dtype=np.uint32)
        b = np.array([0x0F, 0xF0, 0xFF], dtype=np.uint32)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.bitwise_and(ra, rb), np.bitwise_and(a, b))
        assert_eq(rp.bitwise_or(ra, rb), np.bitwise_or(a, b))
        assert_eq(rp.bitwise_xor(ra, rb), np.bitwise_xor(a, b))

    def test_shift_zero(self):
        """Test shifting by zero."""
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        shift = np.int64(0)
        ra = rp.asarray(a)
        r_shift = rp.asarray(shift)
        assert_eq(rp.left_shift(ra, r_shift), np.left_shift(a, shift))
        assert_eq(rp.right_shift(ra, r_shift), np.right_shift(a, shift))

    def test_uint_overflow(self):
        """Test unsigned integer behavior (no overflow in shift)."""
        # Large unsigned values
        a = np.array([2**31, 2**32 - 1], dtype=np.uint64)
        ra = rp.asarray(a)
        assert_eq(rp.bitwise_not(ra), np.bitwise_not(a))

    def test_signed_negative(self):
        """Test bitwise ops with negative signed integers."""
        a = np.array([-1, -2, -4, -8], dtype=np.int64)
        b = np.array([0xFF, 0xFF, 0xFF, 0xFF], dtype=np.int64)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.bitwise_and(ra, rb), np.bitwise_and(a, b))
        assert_eq(rp.bitwise_or(ra, rb), np.bitwise_or(a, b))
        assert_eq(rp.bitwise_not(ra), np.bitwise_not(a))
