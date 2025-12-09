//! Reduction methods for PyRumpyArray: sum, mean, var, std, etc.

use pyo3::prelude::*;

use super::{
    axis_reduction_with_keepdims, check_axis, scalar_reduction_with_keepdims, PyRumpyArray,
    ReductionResult,
};
use crate::array::DType;

#[pymethods]
impl PyRumpyArray {
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn sum(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                self.inner.sum(),
                keepdims,
                self.inner.dtype(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.sum_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn prod(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                self.inner.prod(),
                keepdims,
                self.inner.dtype(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.prod_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn max(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                self.inner.max(),
                keepdims,
                self.inner.dtype(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.max_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn min(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                self.inner.min(),
                keepdims,
                self.inner.dtype(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.min_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn mean(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                self.inner.mean(),
                keepdims,
                self.inner.dtype(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.mean_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn var(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                self.inner.var(),
                keepdims,
                self.inner.dtype(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.var_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    #[pyo3(signature = (axis=None, keepdims=false))]
    fn std(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                self.inner.std(),
                keepdims,
                self.inner.dtype(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.std_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    /// Central moment of order k.
    #[pyo3(signature = (k, axis=None, keepdims=false))]
    fn moment(&self, k: usize, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                self.inner.moment(k),
                keepdims,
                self.inner.dtype(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.moment_axis(k, ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    /// Skewness (Fisher's definition).
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn skew(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                self.inner.skew(),
                keepdims,
                self.inner.dtype(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.skew_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    /// Kurtosis (Fisher's definition: excess kurtosis, normal = 0).
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn kurtosis(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                self.inner.kurtosis(),
                keepdims,
                self.inner.dtype(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.kurtosis_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn argmax(&self, axis: Option<isize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.argmax() as f64)),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(
                    self.inner.argmax_axis(ax),
                )))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn argmin(&self, axis: Option<isize>) -> PyResult<ReductionResult> {
        match axis {
            None => Ok(ReductionResult::Scalar(self.inner.argmin() as f64)),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(ReductionResult::Array(Self::new(
                    self.inner.argmin_axis(ax),
                )))
            }
        }
    }

    /// Test if all elements evaluate to True.
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn all(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        let val = if self.inner.all() { 1.0 } else { 0.0 };
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                val,
                keepdims,
                DType::bool(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.all_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }

    /// Test if any element evaluates to True.
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn any(&self, axis: Option<isize>, keepdims: bool) -> PyResult<ReductionResult> {
        let val = if self.inner.any() { 1.0 } else { 0.0 };
        match axis {
            None => Ok(scalar_reduction_with_keepdims(
                &self.inner,
                val,
                keepdims,
                DType::bool(),
            )),
            Some(ax) => {
                let ax = super::resolve_axis(ax, self.inner.ndim());
                check_axis(ax, self.inner.ndim())?;
                Ok(axis_reduction_with_keepdims(
                    self.inner.any_axis(ax),
                    ax,
                    keepdims,
                ))
            }
        }
    }
}
