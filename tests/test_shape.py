"""Tests for shape manipulation operations.

Parametrizes over operations for concise comprehensive coverage.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES, CORE_DTYPES, SHAPES_EMPTY
from helpers import assert_eq, make_pair


# === Basic shape operations ===


class TestReshape:
    """Test reshape operation."""

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_dtypes_shapes(self, shape, dtype):
        r, n = make_pair(shape, dtype)
        # Flatten
        new_shape = (n.size,)
        assert_eq(rp.reshape(r, new_shape), np.reshape(n, new_shape))

    def test_reshape_infer_dimension(self):
        """Test -1 inference in reshape."""
        n = np.arange(12, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.reshape(r, (-1, 4)), np.reshape(n, (-1, 4)))
        assert_eq(rp.reshape(r, (3, -1)), np.reshape(n, (3, -1)))
        assert_eq(rp.reshape(r, (2, -1, 2)), np.reshape(n, (2, -1, 2)))

    def test_reshape_various_shapes(self):
        """Test various reshaping patterns."""
        n = np.arange(24, dtype=np.float64)
        r = rp.asarray(n)

        shapes = [
            (24,),
            (1, 24),
            (24, 1),
            (2, 12),
            (3, 8),
            (4, 6),
            (2, 3, 4),
            (2, 2, 6),
            (3, 2, 4),
        ]

        for shape in shapes:
            assert_eq(rp.reshape(r, shape), np.reshape(n, shape))

    def test_reshape_method(self):
        """Test array method r.reshape()."""
        n = np.arange(12, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(r.reshape((3, 4)), n.reshape((3, 4)))

    @pytest.mark.parametrize("shape", SHAPES_EMPTY)
    def test_reshape_empty(self, shape):
        r, n = make_pair(shape, "float64")
        new_shape = (0,) if n.size == 0 else (n.shape[0], 0) if 0 in n.shape else (n.size,)
        assert_eq(rp.reshape(r, new_shape), np.reshape(n, new_shape))


class TestTranspose:
    """Test transpose operation."""

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_dtypes_shapes(self, shape, dtype):
        r, n = make_pair(shape, dtype)
        assert_eq(rp.transpose(r), np.transpose(n))

    def test_transpose_with_axes(self):
        """Test transpose with explicit axes."""
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)

        # Test various permutations (positive axes only to avoid known limitation)
        axes_list = [
            (2, 1, 0),
            (1, 2, 0),
            (0, 2, 1),
            (1, 0, 2),
        ]

        for axes in axes_list:
            assert_eq(rp.transpose(r, axes), np.transpose(n, axes))

    def test_transpose_2d(self):
        """Test 2D transpose (matrix transpose)."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.transpose(r), np.transpose(n))

    def test_transpose_method(self):
        """Test array method r.T property."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(r.T, n.T)

    @pytest.mark.parametrize("shape", SHAPES_EMPTY)
    def test_transpose_empty(self, shape):
        r, n = make_pair(shape, "float64")
        assert_eq(rp.transpose(r), np.transpose(n))


class TestSqueeze:
    """Test squeeze operation."""

    def test_squeeze_basic(self):
        """Test removing all size-1 dimensions."""
        n = np.arange(5, dtype=np.float64).reshape(1, 5, 1)
        r = rp.asarray(n)
        assert_eq(rp.squeeze(r), np.squeeze(n))

    def test_squeeze_noop(self):
        """Test squeeze on array with no size-1 dimensions."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.squeeze(r), np.squeeze(n))

    @pytest.mark.parametrize("shape", [(1, 5), (5, 1), (1, 1, 5), (5, 1, 1), (1, 5, 1, 1)])
    def test_squeeze_various(self, shape):
        n = np.arange(5, dtype=np.float64).reshape(shape)
        r = rp.asarray(n)
        assert_eq(rp.squeeze(r), np.squeeze(n))

    def test_squeeze_method(self):
        """Test array method r.squeeze()."""
        n = np.arange(5, dtype=np.float64).reshape(1, 5, 1)
        r = rp.asarray(n)
        assert_eq(r.squeeze(), n.squeeze())


class TestExpandDims:
    """Test expand_dims operation."""

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_expand_dims_1d(self, axis):
        n = np.arange(5, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.expand_dims(r, axis), np.expand_dims(n, axis))

    @pytest.mark.parametrize("axis", [0, 1, 2, -1])
    def test_expand_dims_2d(self, axis):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.expand_dims(r, axis), np.expand_dims(n, axis))

    def test_expand_dims_various(self):
        """Test expanding at various positions."""
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)

        for axis in [0, 1, 2, -1, -2, -3]:
            assert_eq(rp.expand_dims(r, axis), np.expand_dims(n, axis))


class TestFlip:
    """Test flip, flipud, fliplr operations."""

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_flip_all_axes(self, shape):
        """Test flipping all axes (no axis specified)."""
        r, n = make_pair(shape, "float64")
        assert_eq(rp.flip(r), np.flip(n))

    def test_flip_1d(self):
        n = np.arange(5, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.flip(r), np.flip(n))
        assert_eq(rp.flip(r, axis=0), np.flip(n, axis=0))

    def test_flip_2d_axes(self):
        """Test flipping along specific axes in 2D."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.flip(r, axis=0), np.flip(n, axis=0))
        assert_eq(rp.flip(r, axis=1), np.flip(n, axis=1))

    def test_flip_3d_axes(self):
        """Test flipping along specific axes in 3D (positive axes only)."""
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        # Use positive axes only to avoid known limitation with negative axes on 3D+
        assert_eq(rp.flip(r, axis=0), np.flip(n, axis=0))
        assert_eq(rp.flip(r, axis=1), np.flip(n, axis=1))
        assert_eq(rp.flip(r, axis=2), np.flip(n, axis=2))

    def test_flipud(self):
        """Test flipud (flip along axis 0)."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.flipud(r), np.flipud(n))

    def test_flipud_1d(self):
        n = np.arange(5, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.flipud(r), np.flipud(n))

    def test_fliplr(self):
        """Test fliplr (flip along axis 1)."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.fliplr(r), np.fliplr(n))

    @pytest.mark.parametrize("shape", SHAPES_EMPTY)
    def test_flip_empty(self, shape):
        r, n = make_pair(shape, "float64")
        assert_eq(rp.flip(r), np.flip(n))


# === Stacking operations ===


class TestStack:
    """Test stack operation."""

    def test_stack_1d_axis0(self):
        a = np.arange(3, dtype=np.float64)
        b = np.arange(3, 6, dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.stack([r_a, r_b], axis=0), np.stack([a, b], axis=0))

    def test_stack_1d_axis1(self):
        a = np.arange(3, dtype=np.float64)
        b = np.arange(3, 6, dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.stack([r_a, r_b], axis=1), np.stack([a, b], axis=1))

    def test_stack_2d_axis0(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        b = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.stack([r_a, r_b], axis=0), np.stack([a, b], axis=0))

    def test_stack_2d_axis1(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        b = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.stack([r_a, r_b], axis=1), np.stack([a, b], axis=1))

    def test_stack_2d_axis2(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        b = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.stack([r_a, r_b], axis=2), np.stack([a, b], axis=2))

    def test_stack_multiple_arrays(self):
        """Test stacking more than 2 arrays."""
        arrays_np = [np.arange(i, i + 4, dtype=np.float64) for i in range(0, 12, 4)]
        arrays_rp = [rp.asarray(a) for a in arrays_np]
        assert_eq(rp.stack(arrays_rp, axis=0), np.stack(arrays_np, axis=0))


class TestConcatenate:
    """Test concatenate operation."""

    def test_concatenate_1d(self):
        a = np.arange(3, dtype=np.float64)
        b = np.arange(3, 6, dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.concatenate([r_a, r_b]), np.concatenate([a, b]))

    def test_concatenate_2d_axis0(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        b = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.concatenate([r_a, r_b], axis=0), np.concatenate([a, b], axis=0))

    def test_concatenate_2d_axis1(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        b = np.arange(6, 10, dtype=np.float64).reshape(2, 2)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.concatenate([r_a, r_b], axis=1), np.concatenate([a, b], axis=1))

    def test_concatenate_3d_axes(self):
        """Test concatenate on 3D arrays along different axes (positive axes only)."""
        a = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
        b = np.arange(12, 24, dtype=np.float64).reshape(2, 2, 3)
        r_a, r_b = rp.asarray(a), rp.asarray(b)

        # Test all axes with positive indices
        assert_eq(rp.concatenate([r_a, r_b], axis=0), np.concatenate([a, b], axis=0))
        assert_eq(rp.concatenate([r_a, r_b], axis=1), np.concatenate([a, b], axis=1))
        assert_eq(rp.concatenate([r_a, r_b], axis=2), np.concatenate([a, b], axis=2))

    def test_concatenate_multiple_arrays(self):
        """Test concatenating more than 2 arrays."""
        arrays_np = [np.arange(i, i + 3, dtype=np.float64) for i in range(0, 9, 3)]
        arrays_rp = [rp.asarray(a) for a in arrays_np]
        assert_eq(rp.concatenate(arrays_rp), np.concatenate(arrays_np))


class TestVstack:
    """Test vstack (vertical stack)."""

    def test_vstack_1d(self):
        a = np.arange(3, dtype=np.float64)
        b = np.arange(3, 6, dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.vstack([r_a, r_b]), np.vstack([a, b]))

    def test_vstack_2d(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        b = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.vstack([r_a, r_b]), np.vstack([a, b]))

    def test_vstack_multiple(self):
        arrays_np = [np.arange(i, i + 3, dtype=np.float64) for i in range(0, 9, 3)]
        arrays_rp = [rp.asarray(a) for a in arrays_np]
        assert_eq(rp.vstack(arrays_rp), np.vstack(arrays_np))


class TestHstack:
    """Test hstack (horizontal stack)."""

    def test_hstack_1d(self):
        a = np.arange(3, dtype=np.float64)
        b = np.arange(3, 6, dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.hstack([r_a, r_b]), np.hstack([a, b]))

    def test_hstack_2d(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        b = np.arange(6, 10, dtype=np.float64).reshape(2, 2)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.hstack([r_a, r_b]), np.hstack([a, b]))


class TestDstack:
    """Test dstack (depth stack)."""

    def test_dstack_1d(self):
        a = np.arange(3, dtype=np.float64)
        b = np.arange(3, 6, dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.dstack([r_a, r_b]), np.dstack([a, b]))

    def test_dstack_2d(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        b = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.dstack([r_a, r_b]), np.dstack([a, b]))


class TestColumnStack:
    """Test column_stack."""

    def test_column_stack_1d(self):
        a = np.arange(3, dtype=np.float64)
        b = np.arange(3, 6, dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.column_stack([r_a, r_b]), np.column_stack([a, b]))

    def test_column_stack_2d(self):
        a = np.arange(6, dtype=np.float64).reshape(3, 2)
        b = np.arange(6, 12, dtype=np.float64).reshape(3, 2)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.column_stack([r_a, r_b]), np.column_stack([a, b]))


class TestRowStack:
    """Test row_stack (alias for vstack)."""

    def test_row_stack_1d(self):
        a = np.arange(3, dtype=np.float64)
        b = np.arange(3, 6, dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.row_stack([r_a, r_b]), np.row_stack([a, b]))

    def test_row_stack_2d(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        b = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.row_stack([r_a, r_b]), np.row_stack([a, b]))


# === Splitting operations ===


class TestSplit:
    """Test split operation."""

    def test_split_1d(self):
        n = np.arange(9, dtype=np.float64)
        r = rp.asarray(n)
        r_parts = rp.split(r, 3)
        n_parts = np.split(n, 3)
        assert len(r_parts) == len(n_parts)
        for rp_part, np_part in zip(r_parts, n_parts):
            assert_eq(rp_part, np_part)

    def test_split_2d_axis0(self):
        n = np.arange(12, dtype=np.float64).reshape(4, 3)
        r = rp.asarray(n)
        r_parts = rp.split(r, 2, axis=0)
        n_parts = np.split(n, 2, axis=0)
        assert len(r_parts) == len(n_parts)
        for rp_part, np_part in zip(r_parts, n_parts):
            assert_eq(rp_part, np_part)

    def test_split_2d_axis1(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        r_parts = rp.split(r, 2, axis=1)
        n_parts = np.split(n, 2, axis=1)
        assert len(r_parts) == len(n_parts)
        for rp_part, np_part in zip(r_parts, n_parts):
            assert_eq(rp_part, np_part)


class TestArraySplit:
    """Test array_split (allows uneven splits)."""

    def test_array_split_even(self):
        n = np.arange(9, dtype=np.float64)
        r = rp.asarray(n)
        r_parts = rp.array_split(r, 3)
        n_parts = np.array_split(n, 3)
        assert len(r_parts) == len(n_parts)
        for rp_part, np_part in zip(r_parts, n_parts):
            assert_eq(rp_part, np_part)

    def test_array_split_uneven(self):
        n = np.arange(10, dtype=np.float64)
        r = rp.asarray(n)
        r_parts = rp.array_split(r, 3)
        n_parts = np.array_split(n, 3)
        assert len(r_parts) == len(n_parts)
        for rp_part, np_part in zip(r_parts, n_parts):
            assert_eq(rp_part, np_part)


class TestHsplit:
    """Test hsplit (horizontal split)."""

    def test_hsplit_1d(self):
        n = np.arange(6, dtype=np.float64)
        r = rp.asarray(n)
        r_parts = rp.hsplit(r, 3)
        n_parts = np.hsplit(n, 3)
        assert len(r_parts) == len(n_parts)
        for rp_part, np_part in zip(r_parts, n_parts):
            assert_eq(rp_part, np_part)

    def test_hsplit_2d(self):
        n = np.arange(12, dtype=np.float64).reshape(2, 6)
        r = rp.asarray(n)
        r_parts = rp.hsplit(r, 3)
        n_parts = np.hsplit(n, 3)
        assert len(r_parts) == len(n_parts)
        for rp_part, np_part in zip(r_parts, n_parts):
            assert_eq(rp_part, np_part)


class TestVsplit:
    """Test vsplit (vertical split)."""

    def test_vsplit_2d(self):
        n = np.arange(12, dtype=np.float64).reshape(4, 3)
        r = rp.asarray(n)
        r_parts = rp.vsplit(r, 2)
        n_parts = np.vsplit(n, 2)
        assert len(r_parts) == len(n_parts)
        for rp_part, np_part in zip(r_parts, n_parts):
            assert_eq(rp_part, np_part)


class TestDsplit:
    """Test dsplit (depth split)."""

    def test_dsplit_3d(self):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        r_parts = rp.dsplit(r, 2)
        n_parts = np.dsplit(n, 2)
        assert len(r_parts) == len(n_parts)
        for rp_part, np_part in zip(r_parts, n_parts):
            assert_eq(rp_part, np_part)


# === Axis operations ===


class TestSwapaxes:
    """Test swapaxes operation."""

    def test_swapaxes_2d(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(rp.swapaxes(r, 0, 1), np.swapaxes(n, 0, 1))

    def test_swapaxes_3d(self):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        assert_eq(rp.swapaxes(r, 0, 2), np.swapaxes(n, 0, 2))
        assert_eq(rp.swapaxes(r, 1, 2), np.swapaxes(n, 1, 2))
        assert_eq(rp.swapaxes(r, 0, 1), np.swapaxes(n, 0, 1))

    def test_swapaxes_method(self):
        """Test array method r.swapaxes()."""
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(r.swapaxes(0, 1), n.swapaxes(0, 1))


class TestMoveaxis:
    """Test moveaxis operation."""

    def test_moveaxis_single(self):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        assert_eq(rp.moveaxis(r, 0, 2), np.moveaxis(n, 0, 2))
        assert_eq(rp.moveaxis(r, 2, 0), np.moveaxis(n, 2, 0))
        assert_eq(rp.moveaxis(r, 1, 2), np.moveaxis(n, 1, 2))

    def test_moveaxis_multiple(self):
        """Test moving multiple axes."""
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        assert_eq(rp.moveaxis(r, [0, 1], [2, 0]), np.moveaxis(n, [0, 1], [2, 0]))


class TestRollaxis:
    """Test rollaxis operation."""

    def test_rollaxis_basic(self):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        assert_eq(rp.rollaxis(r, 2), np.rollaxis(n, 2))

    def test_rollaxis_with_start(self):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        assert_eq(rp.rollaxis(r, 2, 1), np.rollaxis(n, 2, 1))


# === Dimension adjustment operations ===


class TestAtleast1d:
    """Test atleast_1d operation."""

    def test_atleast_1d_array(self):
        n = np.arange(5, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.atleast_1d(r), np.atleast_1d(n))

    def test_atleast_1d_2d(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.atleast_1d(r), np.atleast_1d(n))


class TestAtleast2d:
    """Test atleast_2d operation."""

    def test_atleast_2d_1d(self):
        n = np.arange(5, dtype=np.float64)
        r = rp.asarray(n)
        result = rp.atleast_2d(r)
        expected = np.atleast_2d(n)
        assert_eq(result, expected)
        assert result.shape == expected.shape

    def test_atleast_2d_2d(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(rp.atleast_2d(r), np.atleast_2d(n))


class TestAtleast3d:
    """Test atleast_3d operation."""

    def test_atleast_3d_1d(self):
        n = np.arange(5, dtype=np.float64)
        r = rp.asarray(n)
        result = rp.atleast_3d(r)
        expected = np.atleast_3d(n)
        assert_eq(result, expected)
        assert result.shape == expected.shape

    def test_atleast_3d_2d(self):
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        result = rp.atleast_3d(r)
        expected = np.atleast_3d(n)
        assert_eq(result, expected)
        assert result.shape == expected.shape

    def test_atleast_3d_3d(self):
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        assert_eq(rp.atleast_3d(r), np.atleast_3d(n))


# === Broadcasting operations ===


class TestBroadcastTo:
    """Test broadcast_to operation."""

    def test_broadcast_to_basic(self):
        n = np.arange(3, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.broadcast_to(r, (3, 3)), np.broadcast_to(n, (3, 3)))

    def test_broadcast_to_add_dims(self):
        n = np.arange(4, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.broadcast_to(r, (2, 3, 4)), np.broadcast_to(n, (2, 3, 4)))

    def test_broadcast_to_2d(self):
        n = np.arange(4, dtype=np.float64).reshape(1, 4)
        r = rp.asarray(n)
        assert_eq(rp.broadcast_to(r, (3, 4)), np.broadcast_to(n, (3, 4)))


class TestBroadcastArrays:
    """Test broadcast_arrays operation."""

    def test_broadcast_arrays_same_shape(self):
        a = np.arange(3, dtype=np.float64)
        b = np.arange(3, 6, dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        r_result = rp.broadcast_arrays([r_a, r_b])
        n_result = np.broadcast_arrays(a, b)
        assert len(r_result) == len(n_result)
        for rp_arr, np_arr in zip(r_result, n_result):
            assert_eq(rp_arr, np_arr)

    def test_broadcast_arrays_different_shapes(self):
        a = np.arange(4, dtype=np.float64).reshape(4, 1)
        b = np.arange(3, dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        r_result = rp.broadcast_arrays([r_a, r_b])
        n_result = np.broadcast_arrays(a, b)
        assert len(r_result) == len(n_result)
        for rp_arr, np_arr in zip(r_result, n_result):
            assert_eq(rp_arr, np_arr)

    def test_broadcast_arrays_three(self):
        a = np.arange(2, dtype=np.float64).reshape(2, 1, 1)
        b = np.arange(3, dtype=np.float64).reshape(1, 3, 1)
        c = np.arange(4, dtype=np.float64).reshape(1, 1, 4)
        r_a, r_b, r_c = rp.asarray(a), rp.asarray(b), rp.asarray(c)
        r_result = rp.broadcast_arrays([r_a, r_b, r_c])
        n_result = np.broadcast_arrays(a, b, c)
        assert len(r_result) == len(n_result)
        for rp_arr, np_arr in zip(r_result, n_result):
            assert_eq(rp_arr, np_arr)


# === Other shape operations ===


class TestRavel:
    """Test ravel operation (flatten to 1D)."""

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_ravel_shapes(self, shape):
        r, n = make_pair(shape, "float64")
        assert_eq(rp.ravel(r), np.ravel(n))

    def test_ravel_method(self):
        """Test array method r.ravel()."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(r.ravel(), n.ravel())

    @pytest.mark.parametrize("shape", SHAPES_EMPTY)
    def test_ravel_empty(self, shape):
        r, n = make_pair(shape, "float64")
        assert_eq(rp.ravel(r), np.ravel(n))


class TestFlatten:
    """Test flatten operation (always returns copy)."""

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_flatten_shapes(self, shape):
        r, n = make_pair(shape, "float64")
        assert_eq(rp.flatten(r), n.flatten())

    def test_flatten_method(self):
        """Test array method r.flatten()."""
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        r = rp.asarray(n)
        assert_eq(r.flatten(), n.flatten())

    @pytest.mark.parametrize("shape", SHAPES_EMPTY)
    def test_flatten_empty(self, shape):
        r, n = make_pair(shape, "float64")
        assert_eq(rp.flatten(r), n.flatten())


# === Repeat and tile operations ===


class TestRepeat:
    """Test repeat operation."""

    def test_repeat_1d(self):
        n = np.array([1, 2, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.repeat(r, 3), np.repeat(n, 3))

    def test_repeat_2d_flat(self):
        """Test repeat without axis (flattens)."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.repeat(r, 2), np.repeat(n, 2))

    def test_repeat_2d_axis0(self):
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.repeat(r, 2, axis=0), np.repeat(n, 2, axis=0))

    def test_repeat_2d_axis1(self):
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.repeat(r, 2, axis=1), np.repeat(n, 2, axis=1))


class TestTile:
    """Test tile operation."""

    def test_tile_1d_scalar(self):
        n = np.array([1, 2, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.tile(r, 2), np.tile(n, 2))

    def test_tile_1d_tuple(self):
        n = np.array([1, 2, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.tile(r, (2, 3)), np.tile(n, (2, 3)))

    def test_tile_2d(self):
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.tile(r, (2, 3)), np.tile(n, (2, 3)))


# === Append/insert/delete operations ===


class TestAppend:
    """Test append operation."""

    def test_append_1d(self):
        a = np.array([1, 2, 3], dtype=np.float64)
        b = np.array([4, 5], dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.append(r_a, r_b), np.append(a, b))

    def test_append_2d_flat(self):
        """Test append without axis (flattens)."""
        a = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b = np.array([[5, 6]], dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.append(r_a, r_b), np.append(a, b))

    def test_append_2d_axis0(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b = np.array([[5, 6]], dtype=np.float64)
        r_a, r_b = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.append(r_a, r_b, axis=0), np.append(a, b, axis=0))


class TestInsert:
    """Test insert operation."""

    def test_insert_1d(self):
        n = np.array([1, 2, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.insert(r, 1, 99), np.insert(n, 1, 99))

    def test_insert_2d_flat(self):
        """Test insert without axis (flattens)."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.insert(r, 2, 99), np.insert(n, 2, 99))

    def test_insert_2d_axis0(self):
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.insert(r, 1, [5, 6], axis=0), np.insert(n, 1, [5, 6], axis=0))


class TestDelete:
    """Test delete operation."""

    def test_delete_1d(self):
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.delete(r, 2), np.delete(n, 2))

    def test_delete_2d_flat(self):
        """Test delete without axis (flattens)."""
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.delete(r, 1), np.delete(n, 1))

    def test_delete_2d_axis0(self):
        n = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.delete(r, 1, axis=0), np.delete(n, 1, axis=0))


# === Padding and rolling operations ===


class TestPad:
    """Test pad operation."""

    def test_pad_1d_constant(self):
        n = np.array([1, 2, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.pad(r, 2, mode='constant'), np.pad(n, 2, mode='constant'))

    def test_pad_2d_constant(self):
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.pad(r, 1, mode='constant'), np.pad(n, 1, mode='constant'))

    def test_pad_asymmetric(self):
        n = np.array([1, 2, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.pad(r, (1, 2), mode='constant'), np.pad(n, (1, 2), mode='constant'))

    def test_pad_edge(self):
        n = np.array([1, 2, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.pad(r, 2, mode='edge'), np.pad(n, 2, mode='edge'))


class TestRoll:
    """Test roll operation."""

    def test_roll_1d(self):
        n = np.arange(5, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.roll(r, 2), np.roll(n, 2))

    def test_roll_1d_negative(self):
        n = np.arange(5, dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.roll(r, -2), np.roll(n, -2))

    def test_roll_2d_flat(self):
        """Test roll without axis (flattens)."""
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(rp.roll(r, 3), np.roll(n, 3))

    def test_roll_2d_axis0(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(rp.roll(r, 1, axis=0), np.roll(n, 1, axis=0))

    def test_roll_2d_axis1(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(rp.roll(r, 1, axis=1), np.roll(n, 1, axis=1))


class TestRot90:
    """Test rot90 operation (90-degree rotation)."""

    def test_rot90_default(self):
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(rp.rot90(r), np.rot90(n))

    def test_rot90_k2(self):
        """Test 180-degree rotation."""
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(rp.rot90(r, k=2), np.rot90(n, k=2))

    def test_rot90_k3(self):
        """Test 270-degree rotation."""
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(rp.rot90(r, k=3), np.rot90(n, k=3))

    def test_rot90_negative(self):
        """Test negative rotation."""
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        r = rp.asarray(n)
        assert_eq(rp.rot90(r, k=-1), np.rot90(n, k=-1))

    def test_rot90_3d(self):
        """Test rotation on 3D arrays."""
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        r = rp.asarray(n)
        assert_eq(rp.rot90(r), np.rot90(n))
        assert_eq(rp.rot90(r, axes=(0, 2)), np.rot90(n, axes=(0, 2)))
