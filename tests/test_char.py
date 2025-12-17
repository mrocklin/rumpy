"""Tests for string dtype and char module operations."""

import numpy as np
import pytest

import rumpy as rp

# Test fixtures
STRING_ARRAYS = [
    ["hello", "world"],
    ["HELLO", "WORLD"],
    ["Hello", "World"],
    ["", "test"],
    ["  spaces  ", "no_spaces"],
]


class TestStringDtype:
    """Test string array creation and basic operations."""

    def test_create_string_array(self):
        """Test creating string arrays."""
        r = rp.array(["hello", "world"])
        n = np.array(["hello", "world"])

        assert r.shape == n.shape
        assert r.ndim == n.ndim
        assert r.size == n.size

    def test_string_array_different_lengths(self):
        """Test strings of different lengths."""
        r = rp.array(["a", "abc", "abcdefghij"])
        assert r.shape == (3,)
        assert r.size == 3

    def test_string_array_2d(self):
        """Test 2D string arrays."""
        r = rp.array([["hello", "world"], ["foo", "bar"]])
        n = np.array([["hello", "world"], ["foo", "bar"]])

        assert r.shape == n.shape
        assert r.ndim == n.ndim

    def test_empty_strings(self):
        """Test arrays with empty strings."""
        r = rp.array(["", "test", ""])
        assert r.shape == (3,)


class TestCharAddMultiply:
    """Test char.add and char.multiply operations."""

    def test_add_basic(self):
        """Test basic string concatenation."""
        r1 = rp.array(["hello", "foo"])
        r2 = rp.array(["world", "bar"])
        result = rp.char.add(r1, r2)

        n1 = np.array(["hello", "foo"])
        n2 = np.array(["world", "bar"])
        expected = np.char.add(n1, n2)

        assert list(str(result[i]) for i in range(2)) == ["'helloworld'", "'foobar'"]

    def test_multiply_basic(self):
        """Test string repetition."""
        r = rp.array(["ab", "xy"])
        result = rp.char.multiply(r, 3)

        n = np.array(["ab", "xy"])
        expected = np.char.multiply(n, 3)

        # Check results match numpy
        for i in range(2):
            r_str = str(result[i]).strip("'")
            n_str = str(expected[i])
            assert r_str == n_str

    def test_multiply_zero(self):
        """Test multiply by zero."""
        r = rp.array(["hello", "world"])
        result = rp.char.multiply(r, 0)

        n = np.array(["hello", "world"])
        expected = np.char.multiply(n, 0)

        for i in range(2):
            r_str = str(result[i]).strip("'")
            assert r_str == ""


class TestCharCaseConversion:
    """Test case conversion operations."""

    def test_upper(self):
        """Test uppercase conversion."""
        r = rp.array(["hello", "World", "TEST"])
        result = rp.char.upper(r)

        n = np.array(["hello", "World", "TEST"])
        expected = np.char.upper(n)

        for i in range(3):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])

    def test_lower(self):
        """Test lowercase conversion."""
        r = rp.array(["HELLO", "World", "test"])
        result = rp.char.lower(r)

        n = np.array(["HELLO", "World", "test"])
        expected = np.char.lower(n)

        for i in range(3):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])

    def test_capitalize(self):
        """Test capitalize."""
        r = rp.array(["hello", "WORLD", "tEsT"])
        result = rp.char.capitalize(r)

        n = np.array(["hello", "WORLD", "tEsT"])
        expected = np.char.capitalize(n)

        for i in range(3):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])

    def test_title(self):
        """Test title case."""
        r = rp.array(["hello world", "HELLO WORLD", "hELLO wORLD"])
        result = rp.char.title(r)

        n = np.array(["hello world", "HELLO WORLD", "hELLO wORLD"])
        expected = np.char.title(n)

        for i in range(3):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])

    def test_swapcase(self):
        """Test swap case."""
        r = rp.array(["Hello", "WORLD", "test"])
        result = rp.char.swapcase(r)

        n = np.array(["Hello", "WORLD", "test"])
        expected = np.char.swapcase(n)

        for i in range(3):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])


class TestCharStrip:
    """Test strip operations."""

    def test_strip(self):
        """Test strip whitespace."""
        r = rp.array(["  hello  ", "world  ", "  test"])
        result = rp.char.strip(r)

        n = np.array(["  hello  ", "world  ", "  test"])
        expected = np.char.strip(n)

        for i in range(3):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])

    def test_lstrip(self):
        """Test left strip."""
        r = rp.array(["  hello", "world", "  test  "])
        result = rp.char.lstrip(r)

        n = np.array(["  hello", "world", "  test  "])
        expected = np.char.lstrip(n)

        for i in range(3):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])

    def test_rstrip(self):
        """Test right strip."""
        r = rp.array(["hello  ", "world", "  test  "])
        result = rp.char.rstrip(r)

        n = np.array(["hello  ", "world", "  test  "])
        expected = np.char.rstrip(n)

        for i in range(3):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])


class TestCharSearch:
    """Test search operations."""

    def test_find(self):
        """Test find substring."""
        r = rp.array(["hello world", "world hello", "no match"])
        result = rp.char.find(r, "world")

        n = np.array(["hello world", "world hello", "no match"])
        expected = np.char.find(n, "world")

        for i in range(3):
            assert int(result[i]) == int(expected[i])

    def test_find_not_found(self):
        """Test find when substring not found."""
        r = rp.array(["hello", "world"])
        result = rp.char.find(r, "xyz")

        for i in range(2):
            assert int(result[i]) == -1

    def test_rfind(self):
        """Test rfind substring."""
        r = rp.array(["hello hello", "world world", "test"])
        result = rp.char.rfind(r, "hello")

        n = np.array(["hello hello", "world world", "test"])
        expected = np.char.rfind(n, "hello")

        for i in range(3):
            assert int(result[i]) == int(expected[i])

    def test_count(self):
        """Test count substring occurrences."""
        r = rp.array(["hello", "lllll", "world"])
        result = rp.char.count(r, "l")

        n = np.array(["hello", "lllll", "world"])
        expected = np.char.count(n, "l")

        for i in range(3):
            assert int(result[i]) == int(expected[i])

    def test_str_len(self):
        """Test string length."""
        r = rp.array(["hello", "hi", ""])
        result = rp.char.str_len(r)

        n = np.array(["hello", "hi", ""])
        expected = np.char.str_len(n)

        for i in range(3):
            assert int(result[i]) == int(expected[i])


class TestCharReplace:
    """Test replace operation."""

    def test_replace_basic(self):
        """Test basic replace."""
        r = rp.array(["hello world", "world hello"])
        result = rp.char.replace(r, "world", "earth")

        n = np.array(["hello world", "world hello"])
        expected = np.char.replace(n, "world", "earth")

        for i in range(2):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])

    def test_replace_count(self):
        """Test replace with count limit."""
        r = rp.array(["aaa", "aaaa"])
        result = rp.char.replace(r, "a", "b", count=2)

        n = np.array(["aaa", "aaaa"])
        expected = np.char.replace(n, "a", "b", count=2)

        for i in range(2):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])


class TestCharPredicates:
    """Test predicate operations."""

    def test_isalpha(self):
        """Test isalpha."""
        r = rp.array(["hello", "123", "hello123", ""])
        result = rp.char.isalpha(r)

        n = np.array(["hello", "123", "hello123", ""])
        expected = np.char.isalpha(n)

        for i in range(4):
            assert int(result[i]) == int(expected[i])

    def test_isdigit(self):
        """Test isdigit."""
        r = rp.array(["123", "hello", "123hello", ""])
        result = rp.char.isdigit(r)

        n = np.array(["123", "hello", "123hello", ""])
        expected = np.char.isdigit(n)

        for i in range(4):
            assert int(result[i]) == int(expected[i])

    def test_isalnum(self):
        """Test isalnum."""
        r = rp.array(["hello123", "hello", "123", "hello_123", ""])
        result = rp.char.isalnum(r)

        n = np.array(["hello123", "hello", "123", "hello_123", ""])
        expected = np.char.isalnum(n)

        for i in range(5):
            assert int(result[i]) == int(expected[i])

    def test_isupper(self):
        """Test isupper."""
        r = rp.array(["HELLO", "hello", "Hello", "123", ""])
        result = rp.char.isupper(r)

        n = np.array(["HELLO", "hello", "Hello", "123", ""])
        expected = np.char.isupper(n)

        for i in range(5):
            assert int(result[i]) == int(expected[i])

    def test_islower(self):
        """Test islower."""
        r = rp.array(["hello", "HELLO", "Hello", "123", ""])
        result = rp.char.islower(r)

        n = np.array(["hello", "HELLO", "Hello", "123", ""])
        expected = np.char.islower(n)

        for i in range(5):
            assert int(result[i]) == int(expected[i])

    def test_isspace(self):
        """Test isspace."""
        r = rp.array(["   ", "hello", " hello ", ""])
        result = rp.char.isspace(r)

        n = np.array(["   ", "hello", " hello ", ""])
        expected = np.char.isspace(n)

        for i in range(4):
            assert int(result[i]) == int(expected[i])

    def test_istitle(self):
        """Test istitle."""
        r = rp.array(["Hello World", "hello world", "Hello world", "123", ""])
        result = rp.char.istitle(r)

        n = np.array(["Hello World", "hello world", "Hello world", "123", ""])
        expected = np.char.istitle(n)

        for i in range(5):
            assert int(result[i]) == int(expected[i])

    def test_startswith(self):
        """Test startswith."""
        r = rp.array(["hello world", "world hello", "hi there"])
        result = rp.char.startswith(r, "hello")

        n = np.array(["hello world", "world hello", "hi there"])
        expected = np.char.startswith(n, "hello")

        for i in range(3):
            assert int(result[i]) == int(expected[i])

    def test_endswith(self):
        """Test endswith."""
        r = rp.array(["hello world", "world hello", "hi there"])
        result = rp.char.endswith(r, "world")

        n = np.array(["hello world", "world hello", "hi there"])
        expected = np.char.endswith(n, "world")

        for i in range(3):
            assert int(result[i]) == int(expected[i])


class TestCharFormatting:
    """Test formatting operations."""

    def test_center(self):
        """Test center."""
        r = rp.array(["hello", "hi"])
        result = rp.char.center(r, 10)

        n = np.array(["hello", "hi"])
        expected = np.char.center(n, 10)

        for i in range(2):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])

    def test_ljust(self):
        """Test left justify."""
        r = rp.array(["hello", "hi"])
        result = rp.char.ljust(r, 10)

        n = np.array(["hello", "hi"])
        expected = np.char.ljust(n, 10)

        for i in range(2):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])

    def test_rjust(self):
        """Test right justify."""
        r = rp.array(["hello", "hi"])
        result = rp.char.rjust(r, 10)

        n = np.array(["hello", "hi"])
        expected = np.char.rjust(n, 10)

        for i in range(2):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])

    def test_zfill(self):
        """Test zero fill."""
        r = rp.array(["42", "-42", "hello"])
        result = rp.char.zfill(r, 5)

        n = np.array(["42", "-42", "hello"])
        expected = np.char.zfill(n, 5)

        for i in range(3):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])


class TestCharComparison:
    """Test comparison operations."""

    def test_equal(self):
        """Test element-wise equality."""
        r1 = rp.array(["hello", "world", "test"])
        r2 = rp.array(["hello", "WORLD", "test"])
        result = rp.char.equal(r1, r2)

        n1 = np.array(["hello", "world", "test"])
        n2 = np.array(["hello", "WORLD", "test"])
        expected = np.char.equal(n1, n2)

        for i in range(3):
            assert int(result[i]) == int(expected[i])

    def test_not_equal(self):
        """Test element-wise inequality."""
        r1 = rp.array(["hello", "world", "test"])
        r2 = rp.array(["hello", "WORLD", "test"])
        result = rp.char.not_equal(r1, r2)

        n1 = np.array(["hello", "world", "test"])
        n2 = np.array(["hello", "WORLD", "test"])
        expected = np.char.not_equal(n1, n2)

        for i in range(3):
            assert int(result[i]) == int(expected[i])

    def test_less(self):
        """Test element-wise less than."""
        r1 = rp.array(["a", "b", "c"])
        r2 = rp.array(["b", "b", "a"])
        result = rp.char.less(r1, r2)

        n1 = np.array(["a", "b", "c"])
        n2 = np.array(["b", "b", "a"])
        expected = np.char.less(n1, n2)

        for i in range(3):
            assert int(result[i]) == int(expected[i])

    def test_greater(self):
        """Test element-wise greater than."""
        r1 = rp.array(["c", "b", "a"])
        r2 = rp.array(["b", "b", "b"])
        result = rp.char.greater(r1, r2)

        n1 = np.array(["c", "b", "a"])
        n2 = np.array(["b", "b", "b"])
        expected = np.char.greater(n1, n2)

        for i in range(3):
            assert int(result[i]) == int(expected[i])


class TestCharMisc:
    """Test miscellaneous char operations."""

    def test_expandtabs(self):
        """Test expand tabs."""
        r = rp.array(["hello\tworld", "a\tb\tc"])
        result = rp.char.expandtabs(r, 4)

        n = np.array(["hello\tworld", "a\tb\tc"])
        expected = np.char.expandtabs(n, 4)

        for i in range(2):
            r_str = str(result[i]).strip("'")
            assert r_str == str(expected[i])
