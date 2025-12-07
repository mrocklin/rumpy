// Python bindings for random module

use pyo3::prelude::*;
use std::sync::Mutex;

use crate::array::{DType, RumpyArray};
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

    /// Random permutation of integers from 0 to x-1, or a shuffled copy of x.
    fn permutation(&self, x: &Bound<'_, pyo3::PyAny>) -> PyResult<PyRumpyArray> {
        let mut gen = self.inner.lock().unwrap();

        // Try integer first
        if let Ok(n) = x.extract::<usize>() {
            let arr = gen.permutation(n);
            return Ok(PyRumpyArray::new(arr));
        }

        // Try array
        if let Ok(arr_ref) = x.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
            // Copy array data
            let ptr = arr_ref.inner.data_ptr();
            let dtype = arr_ref.inner.dtype();
            let ops = dtype.ops();
            let mut data: Vec<f64> = Vec::with_capacity(arr_ref.inner.size());
            for offset in arr_ref.inner.iter_offsets() {
                data.push(unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0));
            }
            // Shuffle
            gen.shuffle_data(&mut data);
            let result = RumpyArray::from_vec(data, DType::float64());
            return Ok(PyRumpyArray::new(result));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "permutation: x must be an integer or array",
        ))
    }

    /// Shuffle array in place along its first axis.
    fn shuffle(&self, x: &Bound<'_, PyRumpyArray>) -> PyResult<()> {
        let mut gen = self.inner.lock().unwrap();
        let py_arr = x.borrow();

        let arr = &py_arr.inner;
        let n = arr.shape()[0];

        if n <= 1 {
            return Ok(());
        }

        // Get mutable pointer to data
        // Safety: We have exclusive access via PyO3's borrow checking and the GIL
        let ptr = arr.data_ptr() as *mut f64;
        let size = arr.size();
        let data = unsafe { std::slice::from_raw_parts_mut(ptr, size) };

        // For 1D arrays, shuffle elements directly
        if arr.ndim() == 1 {
            for i in (1..n).rev() {
                let j = gen.bounded_uint64((i + 1) as u64) as usize;
                data.swap(i, j);
            }
        } else {
            // For ND arrays, shuffle along first axis by swapping rows
            let row_size = arr.shape()[1..].iter().product::<usize>();
            for i in (1..n).rev() {
                let j = gen.bounded_uint64((i + 1) as u64) as usize;
                // Swap rows i and j
                for k in 0..row_size {
                    data.swap(i * row_size + k, j * row_size + k);
                }
            }
        }

        Ok(())
    }

    /// Generate samples from beta distribution.
    #[pyo3(signature = (a, b, size=None))]
    fn beta(
        &self,
        a: f64,
        b: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    // Return scalar - use beta with shape [1] and extract
                    let arr = gen.beta(a, b, vec![1]);
                    let val = arr.get_element(&[0]);
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.beta(a, b, shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate samples from gamma distribution.
    #[pyo3(signature = (shape_param, scale=1.0, size=None))]
    fn gamma(
        &self,
        shape_param: f64,
        scale: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    let arr = gen.gamma(shape_param, scale, vec![1]);
                    let val = arr.get_element(&[0]);
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.gamma(shape_param, scale, shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate samples from Poisson distribution.
    #[pyo3(signature = (lam=1.0, size=None))]
    fn poisson(
        &self,
        lam: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    let arr = gen.poisson(lam, vec![1]);
                    let val = arr.get_element(&[0]) as i64;
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.poisson(lam, shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate samples from binomial distribution.
    #[pyo3(signature = (n, p, size=None))]
    fn binomial(
        &self,
        n: u64,
        p: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    let arr = gen.binomial(n, p, vec![1]);
                    let val = arr.get_element(&[0]) as i64;
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.binomial(n, p, shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate samples from chi-square distribution.
    #[pyo3(signature = (df, size=None))]
    fn chisquare(
        &self,
        df: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();
            match size {
                None => {
                    let arr = gen.chisquare(df, vec![1]);
                    let val = arr.get_element(&[0]);
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                Some(s) => {
                    let shape = parse_shape(s)?;
                    let arr = gen.chisquare(df, shape);
                    Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
                }
            }
        })
    }

    /// Generate samples from multivariate normal distribution.
    #[pyo3(signature = (mean, cov, size=None))]
    fn multivariate_normal(
        &self,
        mean: Vec<f64>,
        cov: Vec<Vec<f64>>,
        size: Option<usize>,
    ) -> PyResult<PyRumpyArray> {
        let mut gen = self.inner.lock().unwrap();
        let n = size.unwrap_or(1);
        let arr = gen.multivariate_normal(&mean, &cov, n);
        Ok(PyRumpyArray::new(arr))
    }

    /// Generate random samples from a given array or range.
    /// If `a` is an int, samples are from range(a).
    /// If `a` is an array, samples are from the array elements.
    #[pyo3(signature = (a, size=None, replace=true))]
    fn choice(
        &self,
        a: &Bound<'_, pyo3::PyAny>,
        size: Option<&Bound<'_, pyo3::PyAny>>,
        replace: bool,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut gen = self.inner.lock().unwrap();

            // Parse `a` - either integer (range) or array-like
            let values: Option<Vec<f64>> = if a.extract::<usize>().is_ok() {
                None // Integer case: sample indices directly
            } else if let Ok(arr) = a.extract::<pyo3::PyRef<'_, PyRumpyArray>>() {
                let ptr = arr.inner.data_ptr();
                let dtype = arr.inner.dtype();
                let ops = dtype.ops();
                let mut vals = Vec::with_capacity(arr.inner.size());
                for offset in arr.inner.iter_offsets() {
                    vals.push(unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0));
                }
                Some(vals)
            } else if let Ok(list) = a.extract::<Vec<f64>>() {
                Some(list)
            } else if let Ok(list) = a.extract::<Vec<i64>>() {
                Some(list.into_iter().map(|x| x as f64).collect())
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "choice: 'a' must be an int or array-like"
                ));
            };

            let n = match &values {
                Some(v) => v.len(),
                None => a.extract::<usize>().unwrap(),
            };

            if n == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "choice: cannot sample from empty population"
                ));
            }

            let shape = match size {
                None => vec![1],
                Some(s) => parse_shape(s)?,
            };
            let total: usize = shape.iter().product();

            if !replace && total > n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "choice: cannot take more samples than population when replace=False"
                ));
            }

            // Generate samples
            let result_data: Vec<f64> = if replace {
                (0..total).map(|_| {
                    let idx = gen.bounded_uint64(n as u64) as usize;
                    values.as_ref().map_or(idx as f64, |v| v[idx])
                }).collect()
            } else {
                let mut available: Vec<usize> = (0..n).collect();
                (0..total).map(|i| {
                    let j = gen.bounded_uint64((n - i) as u64) as usize;
                    let idx = available.swap_remove(j);
                    values.as_ref().map_or(idx as f64, |v| v[idx])
                }).collect()
            };

            if size.is_none() {
                return Ok(result_data[0].into_pyobject(py)?.into_any().unbind());
            }

            let arr = RumpyArray::from_vec(result_data, DType::float64())
                .reshape(shape)
                .unwrap_or_else(|| RumpyArray::zeros(vec![total], DType::float64()));

            Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind())
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
