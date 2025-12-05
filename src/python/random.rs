// Python bindings for random module

use pyo3::prelude::*;
use std::sync::Mutex;

use crate::random::Generator;
use super::{parse_shape, PyRumpyArray};

/// Python wrapper for Generator.
/// Uses Mutex for interior mutability since Generator needs &mut self
/// and pyo3 requires thread safety.
#[pyclass(name = "Generator")]
pub struct PyGenerator {
    inner: Mutex<Generator>,
}

#[pymethods]
impl PyGenerator {
    /// Create Generator with PCG64DXSM BitGenerator.
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            // Use system time as default seed
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        });
        Self {
            inner: Mutex::new(Generator::new(seed)),
        }
    }

    /// Create Generator with explicit state from numpy's PCG64DXSM.
    /// Extract state from numpy:
    /// ```python
    /// bg = np.random.PCG64DXSM(seed)
    /// state = bg.state['state']['state']
    /// inc = bg.state['state']['inc']
    /// rng = rp.random.Generator.from_numpy_state(state, inc)
    /// ```
    #[staticmethod]
    fn from_numpy_state(state: u128, inc: u128) -> Self {
        Self {
            inner: Mutex::new(Generator::from_numpy_state(state, inc)),
        }
    }

    /// Generate raw u64 value (for testing/debugging).
    fn random_raw(&self) -> u64 {
        self.inner.lock().unwrap().next_u64()
    }

    /// Generate uniform random floats in [0, 1).
    #[pyo3(signature = (size=None))]
    fn random(&self, size: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    // Return scalar
                    Ok(gen.next_f64().into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.random(shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate random integers in [low, high).
    #[pyo3(signature = (low, high=None, size=None))]
    fn integers(
        &self,
        low: i64,
        high: Option<i64>,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        // If high is None, treat low as high and use 0 as low
        let (low, high) = match high {
            Some(h) => (low, h),
            None => (0, low),
        };

        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    // Return scalar - use integers with shape [1] and extract
                    let arr = gen.integers(low, high, vec![1]);
                    let val = arr.get_element(&[0]) as i64;
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.integers(low, high, shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate uniform random floats in [low, high).
    #[pyo3(signature = (low=0.0, high=1.0, size=None))]
    fn uniform(
        &self,
        low: f64,
        high: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    // Return scalar
                    let val = low + gen.next_f64() * (high - low);
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.uniform(low, high, shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate samples from normal distribution.
    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn normal(
        &self,
        loc: f64,
        scale: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    // Return scalar
                    let z = gen.next_standard_normal();
                    Ok((loc + scale * z).into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.normal(loc, scale, shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate samples from standard normal distribution.
    #[pyo3(signature = (size=None))]
    fn standard_normal(&self, size: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    let z = gen.next_standard_normal();
                    Ok(z.into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.standard_normal(shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate samples from exponential distribution.
    #[pyo3(signature = (scale=1.0, size=None))]
    fn exponential(
        &self,
        scale: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    let z = gen.next_standard_exponential();
                    Ok((scale * z).into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.exponential(scale, shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate samples from standard exponential distribution.
    #[pyo3(signature = (size=None))]
    fn standard_exponential(&self, size: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    let z = gen.next_standard_exponential();
                    Ok(z.into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.standard_exponential(shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }
}

/// Create a new Generator with PCG64DXSM BitGenerator.
#[pyfunction]
#[pyo3(signature = (seed=None))]
pub fn default_rng(seed: Option<u64>) -> PyGenerator {
    PyGenerator::new(seed)
}

/// Register random submodule.
pub fn register_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let random = PyModule::new(py, "random")?;

    random.add_class::<PyGenerator>()?;
    random.add_function(wrap_pyfunction!(default_rng, &random)?)?;

    parent.add_submodule(&random)?;

    // Make accessible as rp.random
    py.import("sys")?
        .getattr("modules")?
        .set_item("rumpy.random", &random)?;

    Ok(())
}
