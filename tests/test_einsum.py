"""Tests for einsum and einsum_path.

Einstein summation is a powerful notation for tensor contractions.
NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
"""

import numpy as np
import pytest
import rumpy as rp
from helpers import assert_eq
from conftest import FLOAT_DTYPES


class TestEinsumBasic:
    """Basic einsum operations - dot products, traces, transposes."""

    def test_dot_product(self):
        """i,i-> dot product of two vectors."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.einsum("i,i->", ra, rb), np.einsum("i,i->", a, b))

    def test_sum_all(self):
        """ij-> sum all elements."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)
        assert_eq(rp.einsum("ij->", r), np.einsum("ij->", n))

    def test_sum_axis_0(self):
        """ij->j sum along axis 0."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)
        assert_eq(rp.einsum("ij->j", r), np.einsum("ij->j", n))

    def test_sum_axis_1(self):
        """ij->i sum along axis 1."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)
        assert_eq(rp.einsum("ij->i", r), np.einsum("ij->i", n))

    def test_trace(self):
        """ii-> trace of a matrix."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)
        assert_eq(rp.einsum("ii->", r), np.einsum("ii->", n))

    def test_diagonal(self):
        """ii->i extract diagonal."""
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        r = rp.asarray(n)
        assert_eq(rp.einsum("ii->i", r), np.einsum("ii->i", n))

    def test_transpose(self):
        """ij->ji transpose."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)
        assert_eq(rp.einsum("ij->ji", r), np.einsum("ij->ji", n))

    def test_copy(self):
        """ij->ij explicit copy (identity)."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)
        assert_eq(rp.einsum("ij->ij", r), np.einsum("ij->ij", n))


class TestEinsumMatmul:
    """Matrix multiplication and related operations."""

    def test_matmul_2d(self):
        """ij,jk->ik matrix multiplication."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.einsum("ij,jk->ik", ra, rb), np.einsum("ij,jk->ik", a, b))

    def test_outer_product(self):
        """i,j->ij outer product."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.einsum("i,j->ij", ra, rb), np.einsum("i,j->ij", a, b))

    def test_inner_product(self):
        """i,i-> inner product (same as dot)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.einsum("i,i->", ra, rb), np.einsum("i,i->", a, b))

    def test_hadamard_product(self):
        """ij,ij->ij element-wise multiplication."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.einsum("ij,ij->ij", ra, rb), np.einsum("ij,ij->ij", a, b))

    def test_matrix_vector(self):
        """ij,j->i matrix-vector product."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([5.0, 6.0])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.einsum("ij,j->i", ra, rb), np.einsum("ij,j->i", a, b))

    def test_vector_matrix(self):
        """i,ij->j vector-matrix product."""
        a = np.array([1.0, 2.0])
        b = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.einsum("i,ij->j", ra, rb), np.einsum("i,ij->j", a, b))


class TestEinsumBatch:
    """Batch operations with multiple operands."""

    def test_batch_matmul(self):
        """bij,bjk->bik batch matrix multiplication."""
        a = np.random.randn(3, 4, 5)
        b = np.random.randn(3, 5, 6)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.einsum("bij,bjk->bik", ra, rb), np.einsum("bij,bjk->bik", a, b))

    def test_batch_trace(self):
        """bii->b batch trace."""
        a = np.random.randn(3, 4, 4)
        r = rp.asarray(a)
        assert_eq(rp.einsum("bii->b", r), np.einsum("bii->b", a))

    def test_sum_squares(self):
        """ij,ij-> sum of element-wise product (Frobenius inner product)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(a)
        assert_eq(rp.einsum("ij,ij->", r, r), np.einsum("ij,ij->", a, a))


class TestEinsumChain:
    """Multi-operand contractions."""

    def test_three_way_matmul(self):
        """ij,jk,kl->il chain matrix multiplication."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        c = np.array([[9.0, 10.0], [11.0, 12.0]])
        ra, rb, rc = rp.asarray(a), rp.asarray(b), rp.asarray(c)
        assert_eq(
            rp.einsum("ij,jk,kl->il", ra, rb, rc),
            np.einsum("ij,jk,kl->il", a, b, c),
        )

    def test_tensor_contraction(self):
        """ijk,klj->il tensor contraction."""
        a = np.random.randn(2, 3, 4)
        b = np.random.randn(4, 5, 3)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(
            rp.einsum("ijk,klj->il", ra, rb),
            np.einsum("ijk,klj->il", a, b),
        )


class TestEinsumDtypes:
    """Test einsum preserves dtypes correctly."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dot_product_dtype(self, dtype):
        """Einsum should preserve input dtype."""
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([4, 5, 6], dtype=dtype)
        ra, rb = rp.asarray(a), rp.asarray(b)
        result = rp.einsum("i,i->", ra, rb)
        expected = np.einsum("i,i->", a, b)
        assert_eq(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_matmul_dtype(self, dtype):
        """Matrix multiplication should preserve dtype."""
        a = np.array([[1, 2], [3, 4]], dtype=dtype)
        b = np.array([[5, 6], [7, 8]], dtype=dtype)
        ra, rb = rp.asarray(a), rp.asarray(b)
        result = rp.einsum("ij,jk->ik", ra, rb)
        expected = np.einsum("ij,jk->ik", a, b)
        assert_eq(result, expected)


class TestEinsumImplicit:
    """Test implicit output mode (no -> in subscript)."""

    def test_implicit_trace(self):
        """ii implicit trace (sum repeated indices)."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)
        assert_eq(rp.einsum("ii", r), np.einsum("ii", n))

    def test_implicit_matmul(self):
        """ij,jk implicit matmul (output has union of free indices)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.einsum("ij,jk", ra, rb), np.einsum("ij,jk", a, b))

    def test_implicit_copy(self):
        """ij implicit keeps same indices."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)
        assert_eq(rp.einsum("ij", r), np.einsum("ij", n))


class TestEinsumEdgeCases:
    """Edge cases and special scenarios."""

    def test_empty_array(self):
        """Empty arrays should work."""
        n = np.array([], dtype=np.float64).reshape(0, 3)
        r = rp.asarray(n)
        assert_eq(rp.einsum("ij->j", r), np.einsum("ij->j", n))

    def test_scalar_result(self):
        """Scalar result from contraction."""
        n = np.array([1.0, 2.0, 3.0])
        r = rp.asarray(n)
        result = rp.einsum("i->", r)
        expected = np.einsum("i->", n)
        assert_eq(result, expected)

    def test_single_operand(self):
        """Single operand (copy/transpose/trace)."""
        n = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = rp.asarray(n)
        assert_eq(rp.einsum("ij->", r), np.einsum("ij->", n))

    def test_ones_matrix(self):
        """Test with ones for predictable results."""
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.einsum("ij,jk->ik", ra, rb), np.einsum("ij,jk->ik", a, b))


class TestEinsumOptimize:
    """Test optimize parameter."""

    def test_optimize_true(self):
        """optimize=True should give same result."""
        a = np.random.randn(10, 20)
        b = np.random.randn(20, 30)
        c = np.random.randn(30, 10)
        ra, rb, rc = rp.asarray(a), rp.asarray(b), rp.asarray(c)
        assert_eq(
            rp.einsum("ij,jk,kl->il", ra, rb, rc, optimize=True),
            np.einsum("ij,jk,kl->il", a, b, c, optimize=True),
        )

    def test_optimize_false(self):
        """optimize=False should give same result."""
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 5)
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(
            rp.einsum("ij,jk->ik", ra, rb, optimize=False),
            np.einsum("ij,jk->ik", a, b, optimize=False),
        )


class TestEinsumPath:
    """Test einsum_path function."""

    def test_basic_path(self):
        """einsum_path returns path and info."""
        a = np.ones((3, 4))
        b = np.ones((4, 5))
        ra, rb = rp.asarray(a), rp.asarray(b)
        path, info = rp.einsum_path("ij,jk->ik", ra, rb)
        np_path, np_info = np.einsum_path("ij,jk->ik", a, b)
        # Path format: ['einsum_path', (0, 1), ...] for each contraction pair
        assert path[0] == "einsum_path"
        assert isinstance(info, str)

    def test_path_three_operands(self):
        """Path optimization for three operands."""
        a = np.ones((10, 20))
        b = np.ones((20, 30))
        c = np.ones((30, 10))
        ra, rb, rc = rp.asarray(a), rp.asarray(b), rp.asarray(c)
        path, info = rp.einsum_path("ij,jk,kl->il", ra, rb, rc, optimize="greedy")
        assert path[0] == "einsum_path"
        # Should have two contractions: (a,b), then result with c
        assert len(path) == 3  # 'einsum_path' + 2 tuples

    def test_path_optimal(self):
        """Optimal path should work."""
        a = np.ones((5, 6))
        b = np.ones((6, 7))
        c = np.ones((7, 5))
        ra, rb, rc = rp.asarray(a), rp.asarray(b), rp.asarray(c)
        path, _ = rp.einsum_path("ij,jk,kl->il", ra, rb, rc, optimize="optimal")
        assert path[0] == "einsum_path"
