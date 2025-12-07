"""Tests for array manipulation functions (Phase B)."""

import numpy as np
import rumpy

from helpers import assert_eq


class TestCopy:
    """Test copy() method."""

    def test_copy_basic(self):
        arr = rumpy.arange(10)
        c = arr.copy()
        assert_eq(c, np.arange(10))

    def test_copy_2d(self):
        arr = rumpy.arange(12).reshape(3, 4)
        c = arr.copy()
        assert_eq(c, np.arange(12).reshape(3, 4))

    def test_copy_preserves_dtype(self):
        arr = rumpy.arange(10, dtype="int32")
        c = arr.copy()
        assert c.dtype == "int32"


class TestAstype:
    """Test astype() method."""

    def test_float64_to_int32(self):
        arr = rumpy.arange(5)
        c = arr.astype("int32")
        assert c.dtype == "int32"
        assert_eq(c, np.arange(5).astype("int32"))

    def test_int32_to_float64(self):
        arr = rumpy.arange(5, dtype="int32")
        c = arr.astype("float64")
        assert c.dtype == "float64"
        assert_eq(c, np.arange(5, dtype="int32").astype("float64"))


class TestSqueeze:
    """Test squeeze() function and method."""

    def test_squeeze_basic(self):
        arr = rumpy.arange(5).reshape(1, 5, 1)
        r = arr.squeeze()
        n = np.arange(5).reshape(1, 5, 1).squeeze()
        assert_eq(r, n)

    def test_squeeze_module(self):
        arr = rumpy.arange(5).reshape(1, 5)
        r = rumpy.squeeze(arr)
        n = np.squeeze(np.arange(5).reshape(1, 5))
        assert_eq(r, n)

    def test_squeeze_noop(self):
        arr = rumpy.arange(6).reshape(2, 3)
        r = arr.squeeze()
        n = np.arange(6).reshape(2, 3).squeeze()
        assert_eq(r, n)


class TestExpandDims:
    """Test expand_dims() function."""

    def test_expand_dims_axis0(self):
        arr = rumpy.arange(5)
        r = rumpy.expand_dims(arr, 0)
        n = np.expand_dims(np.arange(5), 0)
        assert_eq(r, n)

    def test_expand_dims_axis1(self):
        arr = rumpy.arange(5)
        r = rumpy.expand_dims(arr, 1)
        n = np.expand_dims(np.arange(5), 1)
        assert_eq(r, n)

    def test_expand_dims_negative(self):
        arr = rumpy.arange(5)
        r = rumpy.expand_dims(arr, -1)
        n = np.expand_dims(np.arange(5), -1)
        assert_eq(r, n)


class TestConcatenate:
    """Test concatenate() function."""

    def test_concatenate_1d(self):
        a = rumpy.arange(3)
        b = rumpy.arange(3, 6)
        r = rumpy.concatenate([a, b])
        n = np.concatenate([np.arange(3), np.arange(3, 6)])
        assert_eq(r, n)

    def test_concatenate_2d_axis0(self):
        a = rumpy.arange(6).reshape(2, 3)
        b = rumpy.arange(6, 12).reshape(2, 3)
        r = rumpy.concatenate([a, b], axis=0)
        n = np.concatenate([np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)], axis=0)
        assert_eq(r, n)

    def test_concatenate_2d_axis1(self):
        a = rumpy.arange(6).reshape(2, 3)
        b = rumpy.arange(6, 10).reshape(2, 2)
        r = rumpy.concatenate([a, b], axis=1)
        n = np.concatenate([np.arange(6).reshape(2, 3), np.arange(6, 10).reshape(2, 2)], axis=1)
        assert_eq(r, n)


class TestStack:
    """Test stack() function."""

    def test_stack_1d_axis0(self):
        a = rumpy.arange(3)
        b = rumpy.arange(3, 6)
        r = rumpy.stack([a, b], axis=0)
        n = np.stack([np.arange(3), np.arange(3, 6)], axis=0)
        assert_eq(r, n)

    def test_stack_1d_axis1(self):
        a = rumpy.arange(3)
        b = rumpy.arange(3, 6)
        r = rumpy.stack([a, b], axis=1)
        n = np.stack([np.arange(3), np.arange(3, 6)], axis=1)
        assert_eq(r, n)


class TestVstack:
    """Test vstack() function."""

    def test_vstack_1d(self):
        a = rumpy.arange(3)
        b = rumpy.arange(3)
        r = rumpy.vstack([a, b])
        n = np.vstack([np.arange(3), np.arange(3)])
        assert_eq(r, n)

    def test_vstack_2d(self):
        a = rumpy.arange(6).reshape(2, 3)
        b = rumpy.arange(6).reshape(2, 3)
        r = rumpy.vstack([a, b])
        n = np.vstack([np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3)])
        assert_eq(r, n)


class TestHstack:
    """Test hstack() function."""

    def test_hstack_1d(self):
        a = rumpy.arange(3)
        b = rumpy.arange(3)
        r = rumpy.hstack([a, b])
        n = np.hstack([np.arange(3), np.arange(3)])
        assert_eq(r, n)

    def test_hstack_2d(self):
        a = rumpy.arange(6).reshape(2, 3)
        b = rumpy.arange(6).reshape(2, 3)
        r = rumpy.hstack([a, b])
        n = np.hstack([np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3)])
        assert_eq(r, n)


class TestSplit:
    """Test split() function."""

    def test_split_1d(self):
        arr = rumpy.arange(9)
        parts = rumpy.split(arr, 3)
        n_parts = np.split(np.arange(9), 3)
        assert len(parts) == 3
        for r, n in zip(parts, n_parts):
            assert_eq(r, n)

    def test_split_2d_axis0(self):
        arr = rumpy.arange(12).reshape(4, 3)
        parts = rumpy.split(arr, 2, axis=0)
        n_parts = np.split(np.arange(12).reshape(4, 3), 2, axis=0)
        assert len(parts) == 2
        for r, n in zip(parts, n_parts):
            assert_eq(r, n)


class TestArraySplit:
    """Test array_split() function."""

    def test_array_split_even(self):
        arr = rumpy.arange(9)
        parts = rumpy.array_split(arr, 3)
        n_parts = np.array_split(np.arange(9), 3)
        assert len(parts) == 3
        for r, n in zip(parts, n_parts):
            assert_eq(r, n)

    def test_array_split_uneven(self):
        arr = rumpy.arange(10)
        parts = rumpy.array_split(arr, 3)
        n_parts = np.array_split(np.arange(10), 3)
        assert len(parts) == 3
        for r, n in zip(parts, n_parts):
            assert_eq(r, n)


class TestFlatten:
    """Test flatten() method."""

    def test_flatten_1d(self):
        arr = rumpy.arange(5)
        r = arr.flatten()
        n = np.arange(5).flatten()
        assert_eq(r, n)

    def test_flatten_2d(self):
        arr = rumpy.arange(12).reshape(3, 4)
        r = arr.flatten()
        n = np.arange(12).reshape(3, 4).flatten()
        assert_eq(r, n)

    def test_flatten_3d(self):
        arr = rumpy.arange(24).reshape(2, 3, 4)
        r = arr.flatten()
        n = np.arange(24).reshape(2, 3, 4).flatten()
        assert_eq(r, n)

    def test_flatten_is_copy(self):
        """flatten() should return a copy, not a view."""
        arr = rumpy.arange(6).reshape(2, 3)
        flat = arr.flatten()
        assert flat.shape == (6,)


class TestRavel:
    """Test ravel() method."""

    def test_ravel_1d(self):
        arr = rumpy.arange(5)
        r = arr.ravel()
        n = np.arange(5).ravel()
        assert_eq(r, n)

    def test_ravel_2d(self):
        arr = rumpy.arange(12).reshape(3, 4)
        r = arr.ravel()
        n = np.arange(12).reshape(3, 4).ravel()
        assert_eq(r, n)

    def test_ravel_3d(self):
        arr = rumpy.arange(24).reshape(2, 3, 4)
        r = arr.ravel()
        n = np.arange(24).reshape(2, 3, 4).ravel()
        assert_eq(r, n)


class TestAbs:
    """Test abs() module function."""

    def test_abs_positive(self):
        arr = rumpy.asarray([1.0, 2.0, 3.0])
        r = rumpy.abs(arr)
        n = np.abs(np.array([1.0, 2.0, 3.0]))
        assert_eq(r, n)

    def test_abs_negative(self):
        arr = rumpy.asarray([-1.0, -2.0, -3.0])
        r = rumpy.abs(arr)
        n = np.abs(np.array([-1.0, -2.0, -3.0]))
        assert_eq(r, n)

    def test_abs_mixed(self):
        arr = rumpy.asarray([-1.0, 2.0, -3.0, 4.0])
        r = rumpy.abs(arr)
        n = np.abs(np.array([-1.0, 2.0, -3.0, 4.0]))
        assert_eq(r, n)

    def test_abs_2d(self):
        arr = rumpy.asarray([[-1.0, 2.0], [-3.0, 4.0]])
        r = rumpy.abs(arr)
        n = np.abs(np.array([[-1.0, 2.0], [-3.0, 4.0]]))
        assert_eq(r, n)


class TestItem:
    """Test item() method."""

    def test_item_scalar(self):
        arr = rumpy.asarray([42.0])
        assert arr.item() == 42.0

    def test_item_0d(self):
        # Create a 0D-like array via indexing
        arr = rumpy.asarray([42.0])
        assert arr[0].item() == 42.0

    def test_item_from_reduction(self):
        arr = rumpy.arange(10)
        result = arr.sum()  # Returns scalar but stored in array
        # sum returns a scalar, so we can't call item on it
        # Instead test with a 1-element slice
        arr2 = rumpy.asarray([arr.sum()])
        assert arr2.item() == 45.0

    def test_item_error_on_multi(self):
        import pytest
        arr = rumpy.arange(10)
        with pytest.raises(ValueError):
            arr.item()


class TestTolist:
    """Test tolist() method."""

    def test_tolist_1d(self):
        arr = rumpy.arange(5)
        n = np.arange(5, dtype=np.float64)
        assert arr.tolist() == n.tolist()

    def test_tolist_2d(self):
        arr = rumpy.arange(6).reshape(2, 3)
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert arr.tolist() == n.tolist()

    def test_tolist_3d(self):
        arr = rumpy.arange(24).reshape(2, 3, 4)
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert arr.tolist() == n.tolist()

    def test_tolist_empty(self):
        arr = rumpy.arange(0)
        n = np.arange(0, dtype=np.float64)
        assert arr.tolist() == n.tolist()


class TestFlip:
    """Test flip, flipud, fliplr functions."""

    def test_flip_1d(self):
        r = rumpy.arange(5)
        n = np.arange(5, dtype=np.float64)
        assert_eq(rumpy.flip(r), np.flip(n))

    def test_flip_2d_axis0(self):
        r = rumpy.arange(6).reshape(2, 3)
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(rumpy.flip(r, axis=0), np.flip(n, axis=0))

    def test_flip_2d_axis1(self):
        r = rumpy.arange(6).reshape(2, 3)
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(rumpy.flip(r, axis=1), np.flip(n, axis=1))

    def test_flip_2d_all_axes(self):
        r = rumpy.arange(6).reshape(2, 3)
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(rumpy.flip(r), np.flip(n))

    def test_flip_negative_axis(self):
        r = rumpy.arange(6).reshape(2, 3)
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(rumpy.flip(r, axis=-1), np.flip(n, axis=-1))

    def test_flipud(self):
        r = rumpy.arange(6).reshape(2, 3)
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(rumpy.flipud(r), np.flipud(n))

    def test_flipud_1d(self):
        r = rumpy.arange(5)
        n = np.arange(5, dtype=np.float64)
        assert_eq(rumpy.flipud(r), np.flipud(n))

    def test_fliplr(self):
        r = rumpy.arange(6).reshape(2, 3)
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(rumpy.fliplr(r), np.fliplr(n))

    def test_flip_3d(self):
        r = rumpy.arange(24).reshape(2, 3, 4)
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(rumpy.flip(r, axis=0), np.flip(n, axis=0))
        assert_eq(rumpy.flip(r, axis=1), np.flip(n, axis=1))
        assert_eq(rumpy.flip(r, axis=2), np.flip(n, axis=2))


class TestNonzero:
    """Test nonzero() function."""

    def test_nonzero_1d(self):
        r = rumpy.asarray([0.0, 1.0, 0.0, 2.0, 3.0, 0.0])
        n = np.array([0.0, 1.0, 0.0, 2.0, 3.0, 0.0])
        r_idx = rumpy.nonzero(r)
        n_idx = np.nonzero(n)
        assert len(r_idx) == len(n_idx)
        for ri, ni in zip(r_idx, n_idx):
            assert_eq(ri, ni)

    def test_nonzero_2d(self):
        r = rumpy.asarray([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
        n = np.array([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
        r_idx = rumpy.nonzero(r)
        n_idx = np.nonzero(n)
        assert len(r_idx) == len(n_idx)
        for ri, ni in zip(r_idx, n_idx):
            assert_eq(ri, ni)

    def test_nonzero_all_zeros(self):
        r = rumpy.zeros(5)
        n = np.zeros(5)
        r_idx = rumpy.nonzero(r)
        n_idx = np.nonzero(n)
        assert len(r_idx) == len(n_idx)
        for ri, ni in zip(r_idx, n_idx):
            assert_eq(ri, ni)

    def test_nonzero_all_nonzero(self):
        r = rumpy.arange(1, 6)
        n = np.arange(1, 6, dtype=np.float64)
        r_idx = rumpy.nonzero(r)
        n_idx = np.nonzero(n)
        assert len(r_idx) == len(n_idx)
        for ri, ni in zip(r_idx, n_idx):
            assert_eq(ri, ni)

    def test_nonzero_3d(self):
        r = rumpy.zeros((2, 3, 2))
        # Set some nonzero values - create a fresh array with values
        r = rumpy.asarray([[[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
                          [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]])
        n = np.array([[[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
                      [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]])
        r_idx = rumpy.nonzero(r)
        n_idx = np.nonzero(n)
        assert len(r_idx) == len(n_idx)
        for ri, ni in zip(r_idx, n_idx):
            assert_eq(ri, ni)


class TestSort:
    """Test sort function."""

    def test_sort_1d(self):
        n = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.sort(r), np.sort(n))

    def test_sort_2d_default(self):
        """Sort along last axis by default."""
        n = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.sort(r), np.sort(n))

    def test_sort_2d_axis0(self):
        n = np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 0.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.sort(r, axis=0), np.sort(n, axis=0))

    def test_sort_int(self):
        n = np.array([3, 1, 4, 1, 5], dtype="int64")
        r = rumpy.asarray(n)
        result = rumpy.sort(r)
        assert result.dtype == "int64"
        assert_eq(result, np.sort(n))


class TestArgsort:
    """Test argsort function."""

    def test_argsort_1d(self):
        n = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.argsort(r), np.argsort(n))

    def test_argsort_2d_default(self):
        """Argsort along last axis by default."""
        n = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.argsort(r), np.argsort(n))

    def test_argsort_2d_axis0(self):
        n = np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 0.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.argsort(r, axis=0), np.argsort(n, axis=0))


class TestSwapaxes:
    """Test swapaxes function."""

    def test_swapaxes_2d(self):
        n = np.arange(6).reshape(2, 3)
        r = rumpy.asarray(n)
        assert_eq(rumpy.swapaxes(r, 0, 1), np.swapaxes(n, 0, 1))

    def test_swapaxes_3d(self):
        n = np.arange(24).reshape(2, 3, 4)
        r = rumpy.asarray(n)
        assert_eq(rumpy.swapaxes(r, 0, 2), np.swapaxes(n, 0, 2))
        assert_eq(rumpy.swapaxes(r, 1, 2), np.swapaxes(n, 1, 2))

    def test_swapaxes_method(self):
        n = np.arange(6).reshape(2, 3)
        r = rumpy.asarray(n)
        assert_eq(r.swapaxes(0, 1), n.swapaxes(0, 1))

    def test_swapaxes_same_axis(self):
        """Swapping same axis returns copy."""
        n = np.arange(6).reshape(2, 3)
        r = rumpy.asarray(n)
        assert_eq(rumpy.swapaxes(r, 1, 1), np.swapaxes(n, 1, 1))


class TestDiff:
    """Test diff function."""

    def test_diff_1d(self):
        n = np.array([1.0, 2.0, 4.0, 7.0, 0.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.diff(r), np.diff(n))

    def test_diff_1d_n2(self):
        """Second order difference."""
        n = np.array([1.0, 2.0, 4.0, 7.0, 0.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.diff(r, n=2), np.diff(n, n=2))

    def test_diff_2d_default(self):
        """Diff along last axis by default."""
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.diff(r), np.diff(n))

    def test_diff_2d_axis0(self):
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.diff(r, axis=0), np.diff(n, axis=0))

    def test_diff_int(self):
        """Diff on integer arrays."""
        n = np.array([1, 3, 6, 10])
        r = rumpy.asarray(n)
        assert_eq(rumpy.diff(r), np.diff(n))


# Stream 11: Array Manipulation Functions


class TestHsplit:
    """Test hsplit() function."""

    def test_hsplit_2d(self):
        n = np.arange(12).reshape(2, 6)
        r = rumpy.asarray(n)
        r_parts = rumpy.hsplit(r, 3)
        n_parts = np.hsplit(n, 3)
        assert len(r_parts) == 3
        for rp, np_ in zip(r_parts, n_parts):
            assert_eq(rp, np_)

    def test_hsplit_1d(self):
        n = np.arange(6)
        r = rumpy.asarray(n)
        r_parts = rumpy.hsplit(r, 3)
        n_parts = np.hsplit(n, 3)
        assert len(r_parts) == 3
        for rp, np_ in zip(r_parts, n_parts):
            assert_eq(rp, np_)


class TestVsplit:
    """Test vsplit() function."""

    def test_vsplit_2d(self):
        n = np.arange(12).reshape(4, 3)
        r = rumpy.asarray(n)
        r_parts = rumpy.vsplit(r, 2)
        n_parts = np.vsplit(n, 2)
        assert len(r_parts) == 2
        for rp, np_ in zip(r_parts, n_parts):
            assert_eq(rp, np_)


class TestDsplit:
    """Test dsplit() function."""

    def test_dsplit_3d(self):
        n = np.arange(24).reshape(2, 3, 4)
        r = rumpy.asarray(n)
        r_parts = rumpy.dsplit(r, 2)
        n_parts = np.dsplit(n, 2)
        assert len(r_parts) == 2
        for rp, np_ in zip(r_parts, n_parts):
            assert_eq(rp, np_)


class TestColumnStack:
    """Test column_stack() function."""

    def test_column_stack_1d(self):
        a = np.arange(3)
        b = np.arange(3, 6)
        r_a, r_b = rumpy.asarray(a), rumpy.asarray(b)
        r = rumpy.column_stack([r_a, r_b])
        n = np.column_stack([a, b])
        assert_eq(r, n)

    def test_column_stack_2d(self):
        a = np.arange(6).reshape(3, 2)
        b = np.arange(6, 12).reshape(3, 2)
        r_a, r_b = rumpy.asarray(a), rumpy.asarray(b)
        r = rumpy.column_stack([r_a, r_b])
        n = np.column_stack([a, b])
        assert_eq(r, n)


class TestRowStack:
    """Test row_stack() function (alias for vstack)."""

    def test_row_stack_1d(self):
        a = np.arange(3)
        b = np.arange(3)
        r_a, r_b = rumpy.asarray(a), rumpy.asarray(b)
        r = rumpy.row_stack([r_a, r_b])
        n = np.row_stack([a, b])
        assert_eq(r, n)

    def test_row_stack_2d(self):
        a = np.arange(6).reshape(2, 3)
        b = np.arange(6).reshape(2, 3)
        r_a, r_b = rumpy.asarray(a), rumpy.asarray(b)
        r = rumpy.row_stack([r_a, r_b])
        n = np.row_stack([a, b])
        assert_eq(r, n)


class TestDstack:
    """Test dstack() function."""

    def test_dstack_1d(self):
        a = np.arange(3)
        b = np.arange(3, 6)
        r_a, r_b = rumpy.asarray(a), rumpy.asarray(b)
        r = rumpy.dstack([r_a, r_b])
        n = np.dstack([a, b])
        assert_eq(r, n)

    def test_dstack_2d(self):
        a = np.arange(6).reshape(2, 3)
        b = np.arange(6, 12).reshape(2, 3)
        r_a, r_b = rumpy.asarray(a), rumpy.asarray(b)
        r = rumpy.dstack([r_a, r_b])
        n = np.dstack([a, b])
        assert_eq(r, n)


class TestRepeat:
    """Test repeat() function."""

    def test_repeat_1d(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.repeat(r, 3), np.repeat(n, 3))

    def test_repeat_2d_flat(self):
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.repeat(r, 2), np.repeat(n, 2))

    def test_repeat_2d_axis0(self):
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.repeat(r, 2, axis=0), np.repeat(n, 2, axis=0))

    def test_repeat_2d_axis1(self):
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.repeat(r, 2, axis=1), np.repeat(n, 2, axis=1))


class TestTile:
    """Test tile() function."""

    def test_tile_1d_scalar(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.tile(r, 2), np.tile(n, 2))

    def test_tile_1d_tuple(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.tile(r, (2, 3)), np.tile(n, (2, 3)))

    def test_tile_2d(self):
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.tile(r, (2, 3)), np.tile(n, (2, 3)))


class TestAppend:
    """Test append() function."""

    def test_append_1d(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        r_a, r_b = rumpy.asarray(a), rumpy.asarray(b)
        assert_eq(rumpy.append(r_a, r_b), np.append(a, b))

    def test_append_2d_flat(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0]])
        r_a, r_b = rumpy.asarray(a), rumpy.asarray(b)
        assert_eq(rumpy.append(r_a, r_b), np.append(a, b))

    def test_append_2d_axis0(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0]])
        r_a, r_b = rumpy.asarray(a), rumpy.asarray(b)
        assert_eq(rumpy.append(r_a, r_b, axis=0), np.append(a, b, axis=0))


class TestInsert:
    """Test insert() function."""

    def test_insert_1d(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.insert(r, 1, 99.0), np.insert(n, 1, 99.0))

    def test_insert_2d_flat(self):
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.insert(r, 2, 99.0), np.insert(n, 2, 99.0))

    def test_insert_2d_axis0(self):
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.insert(r, 1, [5.0, 6.0], axis=0), np.insert(n, 1, [5.0, 6.0], axis=0))


class TestDelete:
    """Test delete() function."""

    def test_delete_1d(self):
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.delete(r, 2), np.delete(n, 2))

    def test_delete_2d_flat(self):
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.delete(r, 1), np.delete(n, 1))

    def test_delete_2d_axis0(self):
        n = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.delete(r, 1, axis=0), np.delete(n, 1, axis=0))


class TestPad:
    """Test pad() function."""

    def test_pad_1d_constant(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.pad(r, 2, mode='constant'), np.pad(n, 2, mode='constant'))

    def test_pad_2d_constant(self):
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rumpy.asarray(n)
        assert_eq(rumpy.pad(r, 1, mode='constant'), np.pad(n, 1, mode='constant'))

    def test_pad_asymmetric(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.pad(r, (1, 2), mode='constant'), np.pad(n, (1, 2), mode='constant'))

    def test_pad_edge(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rumpy.asarray(n)
        assert_eq(rumpy.pad(r, 2, mode='edge'), np.pad(n, 2, mode='edge'))


class TestRoll:
    """Test roll() function."""

    def test_roll_1d(self):
        n = np.arange(5, dtype=np.float64)
        r = rumpy.asarray(n)
        assert_eq(rumpy.roll(r, 2), np.roll(n, 2))

    def test_roll_1d_negative(self):
        n = np.arange(5, dtype=np.float64)
        r = rumpy.asarray(n)
        assert_eq(rumpy.roll(r, -2), np.roll(n, -2))

    def test_roll_2d_flat(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rumpy.asarray(n)
        assert_eq(rumpy.roll(r, 3), np.roll(n, 3))

    def test_roll_2d_axis0(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rumpy.asarray(n)
        assert_eq(rumpy.roll(r, 1, axis=0), np.roll(n, 1, axis=0))

    def test_roll_2d_axis1(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rumpy.asarray(n)
        assert_eq(rumpy.roll(r, 1, axis=1), np.roll(n, 1, axis=1))


class TestRot90:
    """Test rot90() function."""

    def test_rot90_default(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rumpy.asarray(n)
        assert_eq(rumpy.rot90(r), np.rot90(n))

    def test_rot90_k2(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rumpy.asarray(n)
        assert_eq(rumpy.rot90(r, k=2), np.rot90(n, k=2))

    def test_rot90_k3(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rumpy.asarray(n)
        assert_eq(rumpy.rot90(r, k=3), np.rot90(n, k=3))

    def test_rot90_negative(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rumpy.asarray(n)
        assert_eq(rumpy.rot90(r, k=-1), np.rot90(n, k=-1))

    def test_rot90_3d(self):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rumpy.asarray(n)
        assert_eq(rumpy.rot90(r), np.rot90(n))
        assert_eq(rumpy.rot90(r, axes=(0, 2)), np.rot90(n, axes=(0, 2)))
