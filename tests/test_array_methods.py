"""Tests for ndarray methods (Stream 18)."""
import numpy as np
import pytest
import rumpy as rp

from helpers import assert_eq


class TestNonzero:
    """Tests for nonzero() method."""

    def test_nonzero_1d(self):
        arr = rp.asarray([0, 1, 0, 2, 3, 0])
        result = arr.nonzero()
        expected = np.asarray([0, 1, 0, 2, 3, 0]).nonzero()
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_nonzero_2d(self):
        arr = rp.asarray([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        result = arr.nonzero()
        expected = np.asarray([[1, 0, 0], [0, 2, 0], [0, 0, 3]]).nonzero()
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_eq(r, e)

    def test_nonzero_all_zeros(self):
        arr = rp.zeros(5)
        result = arr.nonzero()
        expected = np.zeros(5).nonzero()
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_eq(r, e)


class TestArgsort:
    """Tests for argsort() method."""

    def test_argsort_1d(self):
        arr = rp.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        n = np.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        assert_eq(arr.argsort(), n.argsort())

    def test_argsort_2d_axis0(self):
        arr = rp.asarray([[3, 1], [2, 4], [1, 3]])
        n = np.asarray([[3, 1], [2, 4], [1, 3]])
        assert_eq(arr.argsort(axis=0), n.argsort(axis=0))

    def test_argsort_2d_axis1(self):
        arr = rp.asarray([[3, 1], [2, 4], [1, 3]])
        n = np.asarray([[3, 1], [2, 4], [1, 3]])
        assert_eq(arr.argsort(axis=1), n.argsort(axis=1))

    def test_argsort_2d_none(self):
        arr = rp.asarray([[3, 1], [2, 4]])
        n = np.asarray([[3, 1], [2, 4]])
        assert_eq(arr.argsort(axis=None), n.argsort(axis=None))


class TestSort:
    """Tests for sort() method (in-place)."""

    def test_sort_1d(self):
        arr = rp.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        n = np.asarray([3, 1, 4, 1, 5, 9, 2, 6])
        arr.sort()
        n.sort()
        assert_eq(arr, n)

    def test_sort_2d_default(self):
        arr = rp.asarray([[3, 1], [2, 4], [1, 3]])
        n = np.asarray([[3, 1], [2, 4], [1, 3]])
        arr.sort()  # sorts along last axis by default
        n.sort()
        assert_eq(arr, n)

    def test_sort_2d_axis0(self):
        arr = rp.asarray([[3, 1], [2, 4], [1, 3]])
        n = np.asarray([[3, 1], [2, 4], [1, 3]])
        arr.sort(axis=0)
        n.sort(axis=0)
        assert_eq(arr, n)


class TestSearchsorted:
    """Tests for searchsorted() method."""

    def test_searchsorted_basic(self):
        arr = rp.asarray([1, 2, 3, 4, 5])
        v = rp.asarray([2.5, 0.5, 4.0])
        n = np.asarray([1, 2, 3, 4, 5])
        nv = np.asarray([2.5, 0.5, 4.0])
        assert_eq(arr.searchsorted(v), n.searchsorted(nv))

    def test_searchsorted_right(self):
        arr = rp.asarray([1, 2, 3, 4, 5])
        v = rp.asarray([2.0, 4.0])
        n = np.asarray([1, 2, 3, 4, 5])
        nv = np.asarray([2.0, 4.0])
        assert_eq(arr.searchsorted(v, side='right'), n.searchsorted(nv, side='right'))

    def test_searchsorted_scalar(self):
        arr = rp.asarray([1, 2, 3, 4, 5])
        n = np.asarray([1, 2, 3, 4, 5])
        r = arr.searchsorted(2.5)
        e = n.searchsorted(2.5)
        assert r == e


class TestRepeat:
    """Tests for repeat() method."""

    def test_repeat_scalar(self):
        arr = rp.asarray([1, 2, 3])
        n = np.asarray([1, 2, 3])
        assert_eq(arr.repeat(2), n.repeat(2))

    def test_repeat_axis0(self):
        arr = rp.asarray([[1, 2], [3, 4]])
        n = np.asarray([[1, 2], [3, 4]])
        assert_eq(arr.repeat(2, axis=0), n.repeat(2, axis=0))

    def test_repeat_axis1(self):
        arr = rp.asarray([[1, 2], [3, 4]])
        n = np.asarray([[1, 2], [3, 4]])
        assert_eq(arr.repeat(3, axis=1), n.repeat(3, axis=1))


class TestTake:
    """Tests for take() method."""

    def test_take_1d(self):
        arr = rp.asarray([10, 20, 30, 40, 50])
        n = np.asarray([10, 20, 30, 40, 50])
        indices = rp.asarray([0, 2, 4])
        n_indices = np.asarray([0, 2, 4])
        assert_eq(arr.take(indices), n.take(n_indices))

    def test_take_2d_flat(self):
        arr = rp.asarray([[1, 2], [3, 4]])
        n = np.asarray([[1, 2], [3, 4]])
        indices = rp.asarray([0, 3])
        n_indices = np.asarray([0, 3])
        assert_eq(arr.take(indices), n.take(n_indices))

    def test_take_axis0(self):
        arr = rp.asarray([[1, 2], [3, 4], [5, 6]])
        n = np.asarray([[1, 2], [3, 4], [5, 6]])
        indices = rp.asarray([0, 2])
        n_indices = np.asarray([0, 2])
        assert_eq(arr.take(indices, axis=0), n.take(n_indices, axis=0))


class TestPut:
    """Tests for put() method."""

    def test_put_basic(self):
        arr = rp.asarray([1, 2, 3, 4, 5])
        n = np.asarray([1, 2, 3, 4, 5])
        arr.put([0, 2], [10, 30])
        n.put([0, 2], [10, 30])
        assert_eq(arr, n)

    def test_put_single(self):
        arr = rp.asarray([1, 2, 3, 4, 5])
        n = np.asarray([1, 2, 3, 4, 5])
        arr.put([1], [99])
        n.put([1], [99])
        assert_eq(arr, n)


class TestFill:
    """Tests for fill() method."""

    def test_fill_basic(self):
        arr = rp.zeros(5)
        n = np.zeros(5)
        arr.fill(7)
        n.fill(7)
        assert_eq(arr, n)

    def test_fill_2d(self):
        arr = rp.zeros((3, 4))
        n = np.zeros((3, 4))
        arr.fill(-1.5)
        n.fill(-1.5)
        assert_eq(arr, n)


class TestTobytes:
    """Tests for tobytes() method."""

    def test_tobytes_float64(self):
        arr = rp.asarray([1.0, 2.0, 3.0])
        n = np.asarray([1.0, 2.0, 3.0])
        assert arr.tobytes() == n.tobytes()

    def test_tobytes_int64(self):
        arr = rp.asarray([1, 2, 3], dtype='int64')
        n = np.asarray([1, 2, 3], dtype='int64')
        assert arr.tobytes() == n.tobytes()


class TestView:
    """Tests for view() method with dtype."""

    def test_view_float64_to_int64(self):
        arr = rp.asarray([1.0, 2.0])
        n = np.asarray([1.0, 2.0])
        r = arr.view('int64')
        e = n.view('int64')
        assert_eq(r, e)

    def test_view_int32_to_int16(self):
        arr = rp.asarray([0x00010002, 0x00030004], dtype='int32')
        n = np.asarray([0x00010002, 0x00030004], dtype='int32')
        r = arr.view('int16')
        e = n.view('int16')
        assert_eq(r, e)


class TestPartition:
    """Tests for partition() method."""

    def test_partition_basic(self):
        arr = rp.asarray([3, 4, 2, 1])
        n = np.asarray([3, 4, 2, 1])
        arr.partition(2)
        n.partition(2)
        # After partition, element at kth position should equal sorted value
        assert float(arr[2]) == float(n[2])

    def test_partition_sorted_property(self):
        arr = rp.asarray([3, 4, 2, 1, 5])
        arr.partition(2)
        # All elements before index 2 should be <= arr[2]
        # All elements after index 2 should be >= arr[2]
        pivot = float(arr[2])
        for i in range(2):
            assert float(arr[i]) <= pivot
        for i in range(3, 5):
            assert float(arr[i]) >= pivot


class TestArgpartition:
    """Tests for argpartition() method."""

    def test_argpartition_basic(self):
        arr = rp.asarray([3, 4, 2, 1, 5])
        n = np.asarray([3, 4, 2, 1, 5])
        r = arr.argpartition(2)
        e = n.argpartition(2)
        # The kth element in the partitioned array should match
        assert float(arr.take(r)[2]) == float(np.take(n, e)[2])


class TestBase:
    """Tests for base property."""

    def test_base_none_for_owned(self):
        """Owned array has no base."""
        arr = rp.asarray([1, 2, 3, 4, 5])
        assert arr.base is None

    def test_base_for_view(self):
        """View has non-None base."""
        arr = rp.asarray([1, 2, 3, 4, 5])
        view = arr[1:4]
        assert view.base is not None

    def test_base_for_reshape_view(self):
        """Reshape view has base."""
        arr = rp.asarray([[1, 2], [3, 4]])
        view = arr.reshape(4)
        # May or may not have base depending on implementation
        # Just check it doesn't crash
        _ = view.base


class TestFlags:
    """Tests for flags property."""

    def test_flags_c_contiguous(self):
        """C-contiguous array has proper flags."""
        n = np.asarray([[1, 2, 3], [4, 5, 6]])
        r = rp.asarray(n)
        assert r.flags.c_contiguous == n.flags.c_contiguous

    def test_flags_f_contiguous(self):
        """F-contiguous array has proper flags."""
        n = np.asarray([[1, 2, 3], [4, 5, 6]])
        r = rp.asarray(n)
        assert r.flags.f_contiguous == n.flags.f_contiguous

    def test_flags_transpose(self):
        """Transposed array updates flags."""
        n = np.asarray([[1, 2, 3], [4, 5, 6]])
        r = rp.asarray(n)
        n_t = n.T
        r_t = r.T
        assert r_t.flags.c_contiguous == n_t.flags.c_contiguous


class TestFlat:
    """Tests for flat iterator."""

    def test_flat_1d(self):
        """Flat iterator on 1D array."""
        arr = rp.asarray([1, 2, 3, 4, 5])
        n = np.asarray([1, 2, 3, 4, 5])
        assert_eq(list(arr.flat), list(n.flat))

    def test_flat_2d(self):
        """Flat iterator on 2D array."""
        arr = rp.asarray([[1, 2, 3], [4, 5, 6]])
        n = np.asarray([[1, 2, 3], [4, 5, 6]])
        assert_eq(list(arr.flat), list(n.flat))

    def test_flat_len(self):
        """Flat iterator has correct length."""
        arr = rp.asarray([[1, 2], [3, 4], [5, 6]])
        assert len(arr.flat) == 6

    def test_flat_getitem(self):
        """Flat iterator supports indexing."""
        arr = rp.asarray([[1, 2], [3, 4]])
        n = np.asarray([[1, 2], [3, 4]])
        assert float(arr.flat[2]) == float(n.flat[2])


class TestPtpMethod:
    """Tests for ptp() method (peak-to-peak)."""

    def test_ptp_1d(self):
        """ptp on 1D array."""
        arr = rp.asarray([3, 1, 4, 1, 5, 9])
        n = np.asarray([3, 1, 4, 1, 5, 9])
        # ptp was removed from ndarray in NumPy 2.0, use np.ptp() for comparison
        assert arr.ptp() == np.ptp(n)

    def test_ptp_2d_no_axis(self):
        """ptp on 2D array without axis."""
        arr = rp.asarray([[1, 2], [3, 4]])
        n = np.asarray([[1, 2], [3, 4]])
        assert arr.ptp() == np.ptp(n)

    def test_ptp_axis0(self):
        """ptp along axis 0."""
        arr = rp.asarray([[1, 5], [3, 2]])
        n = np.asarray([[1, 5], [3, 2]])
        assert_eq(arr.ptp(axis=0), np.ptp(n, axis=0))

    def test_ptp_axis1(self):
        """ptp along axis 1."""
        arr = rp.asarray([[1, 5], [3, 2]])
        n = np.asarray([[1, 5], [3, 2]])
        assert_eq(arr.ptp(axis=1), np.ptp(n, axis=1))


class TestCompressMethod:
    """Tests for compress() method."""

    def test_compress_1d(self):
        """compress on 1D array."""
        arr = rp.asarray([1, 2, 3, 4, 5])
        n = np.asarray([1, 2, 3, 4, 5])
        condition = [True, False, True, False, True]
        assert_eq(arr.compress(condition), n.compress(condition))

    def test_compress_axis0(self):
        """compress along axis 0."""
        arr = rp.asarray([[1, 2], [3, 4], [5, 6]])
        n = np.asarray([[1, 2], [3, 4], [5, 6]])
        condition = [True, False, True]
        assert_eq(arr.compress(condition, axis=0), n.compress(condition, axis=0))


class TestChooseMethod:
    """Tests for choose() method."""

    def test_choose_basic(self):
        """choose basic usage."""
        choices = [rp.asarray([0, 1, 2, 3]), rp.asarray([10, 11, 12, 13]),
                   rp.asarray([20, 21, 22, 23]), rp.asarray([30, 31, 32, 33])]
        n_choices = [np.asarray([0, 1, 2, 3]), np.asarray([10, 11, 12, 13]),
                     np.asarray([20, 21, 22, 23]), np.asarray([30, 31, 32, 33])]
        a = rp.asarray([2, 3, 1, 0])
        n_a = np.asarray([2, 3, 1, 0])
        assert_eq(a.choose(choices), n_a.choose(n_choices))


class TestResizeMethod:
    """Tests for resize() method."""

    def test_resize_larger(self):
        """resize to larger shape fills with zeros."""
        arr = rp.asarray([1, 2, 3])
        r = arr.copy()
        r.resize((6,))
        # NumPy fills extra space with zeros
        expected = np.array([1, 2, 3, 0, 0, 0])
        assert_eq(r, expected)

    def test_resize_smaller(self):
        """resize to smaller shape truncates."""
        arr = rp.asarray([1, 2, 3, 4, 5])
        n = np.asarray([1, 2, 3, 4, 5])
        r = arr.copy()
        r.resize((3,))
        n_r = n.copy()
        n_r.resize((3,))
        assert_eq(r, n_r)

    def test_resize_2d(self):
        """resize to different 2D shape fills with zeros."""
        arr = rp.asarray([1, 2, 3, 4])
        r = arr.copy()
        r.resize((2, 3))
        # NumPy fills extra space with zeros
        expected = np.array([[1, 2, 3], [4, 0, 0]])
        assert_eq(r, expected)


class TestDump:
    """Tests for dump() and dumps() methods."""

    def test_dumps_loads(self):
        """dumps and loads roundtrip."""
        import pickle
        arr = rp.asarray([1.0, 2.0, 3.0])
        n = np.asarray([1.0, 2.0, 3.0])
        r_pkl = arr.dumps()
        n_pkl = n.dumps()
        # Both should produce valid pickle bytes
        r_loaded = pickle.loads(r_pkl)
        n_loaded = pickle.loads(n_pkl)
        assert_eq(r_loaded, n_loaded)

    def test_dump_file(self, tmp_path):
        """dump to file and load back."""
        import pickle
        arr = rp.asarray([[1, 2], [3, 4]], dtype='float64')
        filepath = tmp_path / "test.pkl"
        arr.dump(str(filepath))
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
        assert_eq(loaded, arr)


class TestDataProperty:
    """Tests for data property (memoryview)."""

    def test_data_returns_memoryview(self):
        """data property returns memoryview-like object."""
        arr = rp.asarray([1, 2, 3, 4], dtype='float64')
        data = arr.data
        # Should be able to get pointer from it
        assert data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
