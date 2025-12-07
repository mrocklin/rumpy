"""Tests for bitwise operations."""
import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestBitwiseAnd:
    def test_basic(self):
        a = rp.array([0b1100, 0b1010, 0b1111, 0b0000])
        b = rp.array([0b1010, 0b1010, 0b0101, 0b1111])
        r = rp.bitwise_and(a, b)
        n = np.bitwise_and([0b1100, 0b1010, 0b1111, 0b0000], [0b1010, 0b1010, 0b0101, 0b1111])
        assert_eq(r, n)

    def test_broadcasting(self):
        a = rp.array([[0b1111, 0b0000], [0b1010, 0b0101]])
        b = rp.array([0b1100, 0b0011])
        r = rp.bitwise_and(a, b)
        n = np.bitwise_and([[0b1111, 0b0000], [0b1010, 0b0101]], [0b1100, 0b0011])
        assert_eq(r, n)


class TestBitwiseOr:
    def test_basic(self):
        a = rp.array([0b1100, 0b1010, 0b0000])
        b = rp.array([0b0011, 0b0101, 0b0000])
        r = rp.bitwise_or(a, b)
        n = np.bitwise_or([0b1100, 0b1010, 0b0000], [0b0011, 0b0101, 0b0000])
        assert_eq(r, n)


class TestBitwiseXor:
    def test_basic(self):
        a = rp.array([0b1100, 0b1010, 0b1111])
        b = rp.array([0b1010, 0b1010, 0b1111])
        r = rp.bitwise_xor(a, b)
        n = np.bitwise_xor([0b1100, 0b1010, 0b1111], [0b1010, 0b1010, 0b1111])
        assert_eq(r, n)


class TestBitwiseNot:
    def test_basic(self):
        a = rp.array([0, 1, -1, 0b1010])
        r = rp.bitwise_not(a)
        n = np.bitwise_not([0, 1, -1, 0b1010])
        assert_eq(r, n)

    def test_invert_alias(self):
        a = rp.array([0, 1, -1])
        r = rp.invert(a)
        n = np.invert([0, 1, -1])
        assert_eq(r, n)


class TestLeftShift:
    def test_basic(self):
        a = rp.array([1, 2, 4, 8])
        b = rp.array([1, 2, 3, 4])
        r = rp.left_shift(a, b)
        n = np.left_shift([1, 2, 4, 8], [1, 2, 3, 4])
        assert_eq(r, n)

    def test_scalar_shift(self):
        a = rp.array([1, 2, 3, 4])
        b = rp.array([2])
        r = rp.left_shift(a, b)
        n = np.left_shift([1, 2, 3, 4], [2])
        assert_eq(r, n)


class TestRightShift:
    def test_basic(self):
        a = rp.array([16, 32, 64, 128])
        b = rp.array([1, 2, 3, 4])
        r = rp.right_shift(a, b)
        n = np.right_shift([16, 32, 64, 128], [1, 2, 3, 4])
        assert_eq(r, n)

    def test_scalar_shift(self):
        a = rp.array([4, 8, 16, 32])
        b = rp.array([2])
        r = rp.right_shift(a, b)
        n = np.right_shift([4, 8, 16, 32], [2])
        assert_eq(r, n)


class TestOutputDtype:
    def test_bitwise_and_dtype(self):
        r = rp.bitwise_and(rp.array([1, 2]), rp.array([1, 3]))
        assert r.dtype == "int64"

    def test_bitwise_not_dtype(self):
        r = rp.bitwise_not(rp.array([1, 2, 3]))
        assert r.dtype == "int64"

    def test_left_shift_dtype(self):
        r = rp.left_shift(rp.array([1, 2]), rp.array([1, 2]))
        assert r.dtype == "int64"


class TestDunderMethods:
    """Test operator syntax: &, |, ^, ~, <<, >>"""

    def test_and_operator(self):
        a = rp.array([0b1100, 0b1010])
        b = rp.array([0b1010, 0b1010])
        r = a & b
        n = np.array([0b1100, 0b1010]) & np.array([0b1010, 0b1010])
        assert_eq(r, n)

    def test_or_operator(self):
        a = rp.array([0b1100, 0b1010])
        b = rp.array([0b0011, 0b0101])
        r = a | b
        n = np.array([0b1100, 0b1010]) | np.array([0b0011, 0b0101])
        assert_eq(r, n)

    def test_xor_operator(self):
        a = rp.array([0b1100, 0b1010])
        b = rp.array([0b1010, 0b1010])
        r = a ^ b
        n = np.array([0b1100, 0b1010]) ^ np.array([0b1010, 0b1010])
        assert_eq(r, n)

    def test_invert_operator(self):
        a = rp.array([0, 1, -1])
        r = ~a
        n = ~np.array([0, 1, -1])
        assert_eq(r, n)

    def test_lshift_operator(self):
        a = rp.array([1, 2, 4])
        b = rp.array([1, 2, 3])
        r = a << b
        n = np.array([1, 2, 4]) << np.array([1, 2, 3])
        assert_eq(r, n)

    def test_rshift_operator(self):
        a = rp.array([16, 32, 64])
        b = rp.array([1, 2, 3])
        r = a >> b
        n = np.array([16, 32, 64]) >> np.array([1, 2, 3])
        assert_eq(r, n)

    def test_and_with_scalar(self):
        a = rp.array([0b1111, 0b1010, 0b0101])
        r = a & 0b1100
        n = np.array([0b1111, 0b1010, 0b0101]) & 0b1100
        assert_eq(r, n)

    def test_lshift_with_scalar(self):
        a = rp.array([1, 2, 3])
        r = a << 2
        n = np.array([1, 2, 3]) << 2
        assert_eq(r, n)

    def test_rlshift_scalar_array(self):
        a = rp.array([1, 2, 3])
        r = 1 << a
        n = 1 << np.array([1, 2, 3])
        assert_eq(r, n)

    def test_rrshift_scalar_array(self):
        a = rp.array([1, 2, 3])
        r = 64 >> a
        n = 64 >> np.array([1, 2, 3])
        assert_eq(r, n)
