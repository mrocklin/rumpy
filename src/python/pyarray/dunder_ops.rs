//! Dunder arithmetic, comparison, and bitwise operations for PyRumpyArray.

use pyo3::prelude::*;

use super::{
    binary_op_dispatch, binary_op_dispatch_with_elision, bitwise_binary_op_dispatch,
    comparison_op_dispatch, rbitwise_binary_op_dispatch, rbinary_op_dispatch, PyRumpyArray,
};
use crate::ops::bitwise::{bitwise_and, bitwise_or, bitwise_xor, left_shift, right_shift};
use crate::ops::matmul::matmul;
use crate::ops::{BinaryOp, ComparisonOp};

#[pymethods]
impl PyRumpyArray {
    // Binary arithmetic operations
    // Uses elision-aware dispatch for potential in-place buffer reuse

    fn __add__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Add)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Reverse ops don't benefit from elision (self is the non-ephemeral one)
        binary_op_dispatch(&self.inner, other, BinaryOp::Add)
    }

    fn __sub__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Sub)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Sub)
    }

    fn __mul__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Mul)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch(&self.inner, other, BinaryOp::Mul)
    }

    fn __truediv__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Div)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Div)
    }

    fn __pow__(
        slf: PyRef<'_, Self>,
        other: &Bound<'_, PyAny>,
        _modulo: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Pow)
    }

    fn __rpow__(&self, other: &Bound<'_, PyAny>, _modulo: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Pow)
    }

    fn __mod__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::Mod)
    }

    fn __rmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::Mod)
    }

    fn __floordiv__(slf: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op_dispatch_with_elision(slf.as_ptr(), &slf.inner, other, BinaryOp::FloorDiv)
    }

    fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbinary_op_dispatch(&self.inner, other, BinaryOp::FloorDiv)
    }

    fn __matmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyRumpyArray>>() {
            matmul(&self.inner, &other_arr.inner)
                .map(Self::new)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("matmul: incompatible shapes")
                })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "matmul operand must be ndarray",
            ))
        }
    }

    fn __rmatmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_arr) = other.extract::<PyRef<'_, PyRumpyArray>>() {
            matmul(&other_arr.inner, &self.inner)
                .map(Self::new)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("matmul: incompatible shapes")
                })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "matmul operand must be ndarray",
            ))
        }
    }

    // Unary operations

    fn __neg__(&self) -> Self {
        Self::new(self.inner.neg().expect("neg always succeeds"))
    }

    fn __abs__(&self) -> Self {
        Self::new(self.inner.abs().expect("abs always succeeds"))
    }

    fn __float__(&self) -> PyResult<f64> {
        if self.inner.size() != 1 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "only size-1 arrays can be converted to Python scalars",
            ));
        }
        Ok(self.inner.get_element(&vec![0; self.inner.ndim()]))
    }

    fn __int__(&self) -> PyResult<i64> {
        if self.inner.size() != 1 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "only size-1 arrays can be converted to Python scalars",
            ));
        }
        Ok(self.inner.get_element(&vec![0; self.inner.ndim()]) as i64)
    }

    // Comparison operations

    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Gt)
    }

    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Lt)
    }

    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Ge)
    }

    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Le)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Eq)
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        comparison_op_dispatch(&self.inner, other, ComparisonOp::Ne)
    }

    // Bitwise operations

    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        bitwise_binary_op_dispatch(&self.inner, other, bitwise_and)
    }

    fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Commutative
        bitwise_binary_op_dispatch(&self.inner, other, bitwise_and)
    }

    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        bitwise_binary_op_dispatch(&self.inner, other, bitwise_or)
    }

    fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Commutative
        bitwise_binary_op_dispatch(&self.inner, other, bitwise_or)
    }

    fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        bitwise_binary_op_dispatch(&self.inner, other, bitwise_xor)
    }

    fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Commutative
        bitwise_binary_op_dispatch(&self.inner, other, bitwise_xor)
    }

    fn __invert__(&self) -> PyResult<Self> {
        crate::ops::bitwise::bitwise_not(&self.inner)
            .map(Self::new)
            .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("bitwise_not not supported"))
    }

    fn __lshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        bitwise_binary_op_dispatch(&self.inner, other, left_shift)
    }

    fn __rlshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbitwise_binary_op_dispatch(&self.inner, other, left_shift)
    }

    fn __rshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        bitwise_binary_op_dispatch(&self.inner, other, right_shift)
    }

    fn __rrshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        rbitwise_binary_op_dispatch(&self.inner, other, right_shift)
    }
}
