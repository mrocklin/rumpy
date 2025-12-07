"""Tests for Stream 9: Shape Manipulation (Module-level) functions."""
import numpy as np
import rumpy as rp
import pytest
from helpers import assert_eq


class TestReshape:
    def test_reshape_basic(self):
        r = rp.arange(12)
        n = np.arange(12)
        assert_eq(rp.reshape(r, (3, 4)), np.reshape(n, (3, 4)))

    def test_reshape_with_list(self):
        r = rp.arange(12)
        n = np.arange(12)
        assert_eq(rp.reshape(r, [3, 4]), np.reshape(n, [3, 4]))

    def test_reshape_with_infer(self):
        r = rp.arange(12)
        n = np.arange(12)
        assert_eq(rp.reshape(r, (-1, 4)), np.reshape(n, (-1, 4)))
        assert_eq(rp.reshape(r, (3, -1)), np.reshape(n, (3, -1)))

    def test_reshape_flatten(self):
        r = rp.reshape(rp.arange(12), (3, 4))
        n = np.reshape(np.arange(12), (3, 4))
        assert_eq(rp.reshape(r, -1), np.reshape(n, -1))

    def test_reshape_3d(self):
        r = rp.arange(24)
        n = np.arange(24)
        assert_eq(rp.reshape(r, (2, 3, 4)), np.reshape(n, (2, 3, 4)))


class TestRavel:
    def test_ravel_2d(self):
        r = rp.reshape(rp.arange(12), (3, 4))
        n = np.reshape(np.arange(12), (3, 4))
        assert_eq(rp.ravel(r), np.ravel(n))

    def test_ravel_1d(self):
        r = rp.arange(10)
        n = np.arange(10)
        assert_eq(rp.ravel(r), np.ravel(n))

    def test_ravel_3d(self):
        r = rp.reshape(rp.arange(24), (2, 3, 4))
        n = np.reshape(np.arange(24), (2, 3, 4))
        assert_eq(rp.ravel(r), np.ravel(n))


class TestFlatten:
    def test_flatten_2d(self):
        r = rp.reshape(rp.arange(12), (3, 4))
        n = np.reshape(np.arange(12), (3, 4))
        assert_eq(rp.flatten(r), n.flatten())

    def test_flatten_1d(self):
        r = rp.arange(10)
        n = np.arange(10)
        assert_eq(rp.flatten(r), n.flatten())


class TestTranspose:
    def test_transpose_2d(self):
        r = rp.reshape(rp.arange(12), (3, 4))
        n = np.reshape(np.arange(12), (3, 4))
        assert_eq(rp.transpose(r), np.transpose(n))

    def test_transpose_3d(self):
        r = rp.reshape(rp.arange(24), (2, 3, 4))
        n = np.reshape(np.arange(24), (2, 3, 4))
        assert_eq(rp.transpose(r), np.transpose(n))

    def test_transpose_with_axes(self):
        r = rp.reshape(rp.arange(24), (2, 3, 4))
        n = np.reshape(np.arange(24), (2, 3, 4))
        assert_eq(rp.transpose(r, (1, 2, 0)), np.transpose(n, (1, 2, 0)))
        assert_eq(rp.transpose(r, [2, 0, 1]), np.transpose(n, [2, 0, 1]))


class TestAtleast1d:
    def test_atleast_1d_array(self):
        r = rp.arange(5)
        n = np.arange(5)
        # single array - result should match
        r_res = rp.atleast_1d(r)
        n_res = np.atleast_1d(n)
        assert_eq(r_res, n_res)

    def test_atleast_1d_2d(self):
        r = rp.reshape(rp.arange(12), (3, 4))
        n = np.reshape(np.arange(12), (3, 4))
        assert_eq(rp.atleast_1d(r), np.atleast_1d(n))


class TestAtleast2d:
    def test_atleast_2d_1d(self):
        r = rp.arange(5)
        n = np.arange(5)
        r_res = rp.atleast_2d(r)
        n_res = np.atleast_2d(n)
        assert_eq(r_res, n_res)
        assert r_res.shape == n_res.shape

    def test_atleast_2d_2d(self):
        r = rp.reshape(rp.arange(12), (3, 4))
        n = np.reshape(np.arange(12), (3, 4))
        assert_eq(rp.atleast_2d(r), np.atleast_2d(n))


class TestAtleast3d:
    def test_atleast_3d_1d(self):
        r = rp.arange(5)
        n = np.arange(5)
        r_res = rp.atleast_3d(r)
        n_res = np.atleast_3d(n)
        assert_eq(r_res, n_res)
        assert r_res.shape == n_res.shape

    def test_atleast_3d_2d(self):
        r = rp.reshape(rp.arange(12), (3, 4))
        n = np.reshape(np.arange(12), (3, 4))
        r_res = rp.atleast_3d(r)
        n_res = np.atleast_3d(n)
        assert_eq(r_res, n_res)
        assert r_res.shape == n_res.shape

    def test_atleast_3d_3d(self):
        r = rp.reshape(rp.arange(24), (2, 3, 4))
        n = np.reshape(np.arange(24), (2, 3, 4))
        assert_eq(rp.atleast_3d(r), np.atleast_3d(n))


class TestMoveaxis:
    def test_moveaxis_single(self):
        r = rp.reshape(rp.arange(24), (2, 3, 4))
        n = np.reshape(np.arange(24), (2, 3, 4))
        assert_eq(rp.moveaxis(r, 0, -1), np.moveaxis(n, 0, -1))
        assert_eq(rp.moveaxis(r, -1, 0), np.moveaxis(n, -1, 0))

    def test_moveaxis_middle(self):
        r = rp.reshape(rp.arange(24), (2, 3, 4))
        n = np.reshape(np.arange(24), (2, 3, 4))
        assert_eq(rp.moveaxis(r, 1, 2), np.moveaxis(n, 1, 2))

    def test_moveaxis_multiple(self):
        r = rp.reshape(rp.arange(24), (2, 3, 4))
        n = np.reshape(np.arange(24), (2, 3, 4))
        assert_eq(rp.moveaxis(r, [0, 1], [2, 0]), np.moveaxis(n, [0, 1], [2, 0]))


class TestRollaxis:
    def test_rollaxis_basic(self):
        r = rp.reshape(rp.arange(24), (2, 3, 4))
        n = np.reshape(np.arange(24), (2, 3, 4))
        assert_eq(rp.rollaxis(r, 2), np.rollaxis(n, 2))

    def test_rollaxis_with_start(self):
        r = rp.reshape(rp.arange(24), (2, 3, 4))
        n = np.reshape(np.arange(24), (2, 3, 4))
        assert_eq(rp.rollaxis(r, 2, 1), np.rollaxis(n, 2, 1))

    def test_rollaxis_negative(self):
        r = rp.reshape(rp.arange(24), (2, 3, 4))
        n = np.reshape(np.arange(24), (2, 3, 4))
        assert_eq(rp.rollaxis(r, -1), np.rollaxis(n, -1))


class TestBroadcastTo:
    def test_broadcast_to_basic(self):
        r = rp.arange(3)
        n = np.arange(3)
        assert_eq(rp.broadcast_to(r, (3, 3)), np.broadcast_to(n, (3, 3)))

    def test_broadcast_to_add_dims(self):
        r = rp.arange(4)
        n = np.arange(4)
        assert_eq(rp.broadcast_to(r, (2, 3, 4)), np.broadcast_to(n, (2, 3, 4)))

    def test_broadcast_to_2d(self):
        r = rp.reshape(rp.arange(4), (1, 4))
        n = np.reshape(np.arange(4), (1, 4))
        assert_eq(rp.broadcast_to(r, (3, 4)), np.broadcast_to(n, (3, 4)))


class TestBroadcastArrays:
    def test_broadcast_arrays_same_shape(self):
        r1 = rp.arange(3)
        r2 = rp.arange(3)
        n1 = np.arange(3)
        n2 = np.arange(3)

        r_res = rp.broadcast_arrays([r1, r2])
        n_res = np.broadcast_arrays(n1, n2)

        assert len(r_res) == len(n_res)
        for r, n in zip(r_res, n_res):
            assert_eq(r, n)

    def test_broadcast_arrays_different_shapes(self):
        r1 = rp.reshape(rp.arange(4), (4, 1))
        r2 = rp.arange(3)
        n1 = np.reshape(np.arange(4), (4, 1))
        n2 = np.arange(3)

        r_res = rp.broadcast_arrays([r1, r2])
        n_res = np.broadcast_arrays(n1, n2)

        assert len(r_res) == len(n_res)
        for r, n in zip(r_res, n_res):
            assert_eq(r, n)

    def test_broadcast_arrays_3_arrays(self):
        r1 = rp.reshape(rp.arange(2), (2, 1, 1))
        r2 = rp.reshape(rp.arange(3), (1, 3, 1))
        r3 = rp.reshape(rp.arange(4), (1, 1, 4))
        n1 = np.reshape(np.arange(2), (2, 1, 1))
        n2 = np.reshape(np.arange(3), (1, 3, 1))
        n3 = np.reshape(np.arange(4), (1, 1, 4))

        r_res = rp.broadcast_arrays([r1, r2, r3])
        n_res = np.broadcast_arrays(n1, n2, n3)

        assert len(r_res) == len(n_res)
        for r, n in zip(r_res, n_res):
            assert_eq(r, n)


class TestEdgeCases:
    def test_reshape_empty(self):
        r = rp.zeros(0)
        n = np.zeros(0)
        assert_eq(rp.reshape(r, (0, 5)), np.reshape(n, (0, 5)))

    def test_ravel_empty(self):
        r = rp.zeros((0, 3))
        n = np.zeros((0, 3))
        assert_eq(rp.ravel(r), np.ravel(n))

    def test_flatten_empty(self):
        r = rp.zeros((3, 0))
        n = np.zeros((3, 0))
        assert_eq(rp.flatten(r), n.flatten())
