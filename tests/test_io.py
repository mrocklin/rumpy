"""Tests for I/O operations."""
import tempfile
import os
import pytest
import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestLoadtxt:
    """Tests for loadtxt function."""

    def test_simple_text(self, tmp_path):
        """Load simple text file."""
        f = tmp_path / "data.txt"
        f.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")

        n = np.loadtxt(str(f))
        r = rp.loadtxt(str(f))
        assert_eq(r, n)

    def test_with_comments(self, tmp_path):
        """Load text file with comments."""
        f = tmp_path / "data.txt"
        f.write_text("# Header comment\n1.0 2.0\n# Mid comment\n3.0 4.0\n")

        n = np.loadtxt(str(f))
        r = rp.loadtxt(str(f))
        assert_eq(r, n)

    def test_with_delimiter(self, tmp_path):
        """Load CSV file."""
        f = tmp_path / "data.csv"
        f.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n")

        n = np.loadtxt(str(f), delimiter=",")
        r = rp.loadtxt(str(f), delimiter=",")
        assert_eq(r, n)

    def test_skiprows(self, tmp_path):
        """Load file skipping header rows."""
        f = tmp_path / "data.txt"
        f.write_text("Header1\nHeader2\n1.0 2.0\n3.0 4.0\n")

        n = np.loadtxt(str(f), skiprows=2)
        r = rp.loadtxt(str(f), skiprows=2)
        assert_eq(r, n)

    def test_usecols(self, tmp_path):
        """Load specific columns."""
        f = tmp_path / "data.txt"
        f.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")

        n = np.loadtxt(str(f), usecols=[0, 2])
        r = rp.loadtxt(str(f), usecols=[0, 2])
        assert_eq(r, n)

    def test_max_rows(self, tmp_path):
        """Load limited number of rows."""
        f = tmp_path / "data.txt"
        f.write_text("1.0 2.0\n3.0 4.0\n5.0 6.0\n7.0 8.0\n")

        n = np.loadtxt(str(f), max_rows=2)
        r = rp.loadtxt(str(f), max_rows=2)
        assert_eq(r, n)

    def test_dtype(self, tmp_path):
        """Load with specific dtype."""
        f = tmp_path / "data.txt"
        f.write_text("1 2 3\n4 5 6\n")

        n = np.loadtxt(str(f), dtype=np.int32)
        r = rp.loadtxt(str(f), dtype="int32")
        assert_eq(r, n)

    def test_1d_result(self, tmp_path):
        """Load file that results in 1D array."""
        f = tmp_path / "data.txt"
        f.write_text("1.0\n2.0\n3.0\n")

        n = np.loadtxt(str(f))
        r = rp.loadtxt(str(f))
        assert_eq(r, n)


class TestSavetxt:
    """Tests for savetxt function."""

    def test_simple_save(self, tmp_path):
        """Save and load simple array."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)

        nf = tmp_path / "numpy.txt"
        rf = tmp_path / "rumpy.txt"

        np.savetxt(str(nf), n)
        rp.savetxt(str(rf), r)

        # Load both back and compare
        n_loaded = np.loadtxt(str(nf))
        r_loaded = rp.loadtxt(str(rf))
        assert_eq(r_loaded, n_loaded)

    def test_with_delimiter(self, tmp_path):
        """Save CSV format."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)

        nf = tmp_path / "numpy.csv"
        rf = tmp_path / "rumpy.csv"

        np.savetxt(str(nf), n, delimiter=",")
        rp.savetxt(str(rf), r, delimiter=",")

        # Load both back
        n_loaded = np.loadtxt(str(nf), delimiter=",")
        r_loaded = rp.loadtxt(str(rf), delimiter=",")
        assert_eq(r_loaded, n_loaded)

    def test_with_format(self, tmp_path):
        """Save with specific format."""
        n = np.array([[1.5, 2.5], [3.5, 4.5]])
        r = rp.asarray(n)

        rf = tmp_path / "rumpy.txt"
        rp.savetxt(str(rf), r, fmt="%.2f")

        r_loaded = rp.loadtxt(str(rf))
        # Values should be rounded to 2 decimal places
        assert r_loaded[0, 0] == 1.5

    def test_1d_array(self, tmp_path):
        """Save 1D array."""
        n = np.array([1.0, 2.0, 3.0])
        r = rp.asarray(n)

        nf = tmp_path / "numpy.txt"
        rf = tmp_path / "rumpy.txt"

        np.savetxt(str(nf), n)
        rp.savetxt(str(rf), r)

        n_loaded = np.loadtxt(str(nf))
        r_loaded = rp.loadtxt(str(rf))
        assert_eq(r_loaded, n_loaded)


class TestGenfromtxt:
    """Tests for genfromtxt function."""

    def test_simple_load(self, tmp_path):
        """Load simple text file."""
        f = tmp_path / "data.txt"
        f.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")

        n = np.genfromtxt(str(f))
        r = rp.genfromtxt(str(f))
        assert_eq(r, n)

    def test_missing_values(self, tmp_path):
        """Load file with empty values (consistent columns)."""
        f = tmp_path / "data.txt"
        # Use empty values that genfromtxt can handle
        f.write_text("1.0,2.0\n3.0,\n5.0,6.0\n")

        r = rp.genfromtxt(str(f), delimiter=",", filling_values=0.0)
        # Check the shape and that filling worked
        assert r.shape == (3, 2)
        assert r[1, 1] == 0.0  # Empty value filled with 0

    def test_skip_header(self, tmp_path):
        """Load file skipping header."""
        f = tmp_path / "data.txt"
        f.write_text("Header\n1.0 2.0\n3.0 4.0\n")

        n = np.genfromtxt(str(f), skip_header=1)
        r = rp.genfromtxt(str(f), skip_header=1)
        assert_eq(r, n)

    def test_skip_footer(self, tmp_path):
        """Load file skipping footer."""
        f = tmp_path / "data.txt"
        f.write_text("1.0 2.0\n3.0 4.0\nFooter line\n")

        n = np.genfromtxt(str(f), skip_footer=1)
        r = rp.genfromtxt(str(f), skip_footer=1)
        assert_eq(r, n)

    def test_filling_values(self, tmp_path):
        """Load with specific filling value."""
        f = tmp_path / "data.txt"
        f.write_text("1.0,2.0,3.0\n4.0,,6.0\n")

        r = rp.genfromtxt(str(f), delimiter=",", filling_values=-1.0)
        assert r[1, 1] == -1.0


class TestSaveLoad:
    """Tests for save/load (.npy format)."""

    def test_save_load_float64(self, tmp_path):
        """Save and load float64 array."""
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rp.asarray(n)

        rf = tmp_path / "array.npy"
        rp.save(str(rf), r)
        r_loaded = rp.load(str(rf))
        assert_eq(r_loaded, n)

    def test_save_load_int64(self, tmp_path):
        """Save and load int64 array."""
        n = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        r = rp.asarray(n)

        rf = tmp_path / "array.npy"
        rp.save(str(rf), r)
        r_loaded = rp.load(str(rf))
        assert_eq(r_loaded, n)

    def test_save_load_float32(self, tmp_path):
        """Save and load float32 array."""
        n = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        r = rp.asarray(n)

        rf = tmp_path / "array.npy"
        rp.save(str(rf), r)
        r_loaded = rp.load(str(rf))
        assert_eq(r_loaded, n)

    def test_save_load_3d(self, tmp_path):
        """Save and load 3D array."""
        n = np.arange(24).reshape(2, 3, 4).astype(np.float64)
        r = rp.asarray(n)

        rf = tmp_path / "array.npy"
        rp.save(str(rf), r)
        r_loaded = rp.load(str(rf))
        assert_eq(r_loaded, n)

    def test_numpy_interop(self, tmp_path):
        """Save with rumpy, load with numpy."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)

        rf = tmp_path / "array.npy"
        rp.save(str(rf), r)
        n_loaded = np.load(str(rf))
        assert_eq(r, n_loaded)

    def test_numpy_interop_reverse(self, tmp_path):
        """Save with numpy, load with rumpy."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])

        nf = tmp_path / "array.npy"
        np.save(str(nf), n)
        r_loaded = rp.load(str(nf))
        assert_eq(r_loaded, n)

    def test_1d_array(self, tmp_path):
        """Save and load 1D array."""
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = rp.asarray(n)

        rf = tmp_path / "array.npy"
        rp.save(str(rf), r)
        r_loaded = rp.load(str(rf))
        assert_eq(r_loaded, n)


class TestSavezLoad:
    """Tests for savez/savez_compressed and load (.npz format)."""

    def test_savez_load_positional(self, tmp_path):
        """Save and load multiple arrays with positional args."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])
        ra = rp.asarray(a)
        rb = rp.asarray(b)

        rf = tmp_path / "arrays.npz"
        rp.savez(str(rf), ra, rb)
        loaded = rp.load(str(rf))

        assert "arr_0" in loaded
        assert "arr_1" in loaded
        assert_eq(loaded["arr_0"], a)
        assert_eq(loaded["arr_1"], b)

    def test_savez_load_kwargs(self, tmp_path):
        """Save and load arrays with keyword args."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])
        ra = rp.asarray(a)
        rb = rp.asarray(b)

        rf = tmp_path / "arrays.npz"
        rp.savez(str(rf), first=ra, second=rb)
        loaded = rp.load(str(rf))

        assert "first" in loaded
        assert "second" in loaded
        assert_eq(loaded["first"], a)
        assert_eq(loaded["second"], b)

    def test_savez_compressed(self, tmp_path):
        """Save and load compressed arrays."""
        a = np.arange(1000, dtype=np.float64)
        ra = rp.asarray(a)

        rf_compressed = tmp_path / "compressed.npz"
        rf_uncompressed = tmp_path / "uncompressed.npz"

        rp.savez_compressed(str(rf_compressed), arr=ra)
        rp.savez(str(rf_uncompressed), arr=ra)

        # Compressed should be smaller
        assert rf_compressed.stat().st_size < rf_uncompressed.stat().st_size

        # But load the same
        loaded = rp.load(str(rf_compressed))
        assert_eq(loaded["arr"], a)

    def test_numpy_npz_interop(self, tmp_path):
        """Load NumPy .npz with rumpy."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([[4.0, 5.0], [6.0, 7.0]])

        nf = tmp_path / "numpy.npz"
        np.savez(str(nf), first=a, second=b)

        loaded = rp.load(str(nf))
        assert_eq(loaded["first"], a)
        assert_eq(loaded["second"], b)


class TestFrombuffer:
    """Tests for frombuffer function."""

    def test_float64(self):
        """Create array from float64 bytes."""
        n = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        buf = n.tobytes()

        r = rp.frombuffer(buf, dtype="float64")
        assert_eq(r, n)

    def test_int32(self):
        """Create array from int32 bytes."""
        n = np.array([1, 2, 3, 4], dtype=np.int32)
        buf = n.tobytes()

        r = rp.frombuffer(buf, dtype="int32")
        assert_eq(r, n)

    def test_with_offset(self):
        """Create array from buffer with offset."""
        n = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        buf = n.tobytes()

        # Skip first element (8 bytes for float64)
        r = rp.frombuffer(buf, dtype="float64", offset=8)
        expected = np.array([2.0, 3.0, 4.0])
        assert_eq(r, expected)

    def test_with_count(self):
        """Create array from buffer with count limit."""
        n = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        buf = n.tobytes()

        r = rp.frombuffer(buf, dtype="float64", count=2)
        expected = np.array([1.0, 2.0])
        assert_eq(r, expected)


class TestFromfile:
    """Tests for fromfile function."""

    def test_binary_float64(self, tmp_path):
        """Read binary float64 file."""
        n = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        f = tmp_path / "data.bin"
        n.tofile(str(f))

        r = rp.fromfile(str(f), dtype="float64")
        assert_eq(r, n)

    def test_binary_int32(self, tmp_path):
        """Read binary int32 file."""
        n = np.array([1, 2, 3, 4], dtype=np.int32)
        f = tmp_path / "data.bin"
        n.tofile(str(f))

        r = rp.fromfile(str(f), dtype="int32")
        assert_eq(r, n)

    def test_with_count(self, tmp_path):
        """Read limited count from binary file."""
        n = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        f = tmp_path / "data.bin"
        n.tofile(str(f))

        r = rp.fromfile(str(f), dtype="float64", count=2)
        expected = np.array([1.0, 2.0])
        assert_eq(r, expected)

    def test_text_mode(self, tmp_path):
        """Read text file with separator."""
        f = tmp_path / "data.txt"
        f.write_text("1.0,2.0,3.0,4.0")

        r = rp.fromfile(str(f), dtype="float64", sep=",")
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        assert_eq(r, expected)


class TestTofile:
    """Tests for tofile array method."""

    def test_binary_write(self, tmp_path):
        """Write array to binary file."""
        n = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        r = rp.asarray(n)

        rf = tmp_path / "rumpy.bin"
        r.tofile(str(rf))

        # Read back
        r_loaded = rp.fromfile(str(rf), dtype="float64")
        assert_eq(r_loaded, n)

    def test_text_write(self, tmp_path):
        """Write array to text file."""
        n = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        r = rp.asarray(n)

        rf = tmp_path / "rumpy.txt"
        r.tofile(str(rf), sep=",")

        # Read back
        r_loaded = rp.fromfile(str(rf), dtype="float64", sep=",")
        assert_eq(r_loaded, n)

    def test_2d_binary(self, tmp_path):
        """Write 2D array to binary file."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        r = rp.asarray(n)

        rf = tmp_path / "rumpy.bin"
        r.tofile(str(rf))

        # Read back as flat
        r_loaded = rp.fromfile(str(rf), dtype="float64")
        assert_eq(r_loaded, n.ravel())
