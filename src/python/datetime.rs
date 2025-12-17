//! Datetime operations for rumpy.
//!
//! Implements numpy datetime functions like isnat, datetime_as_string, datetime_data,
//! and business day functions.

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::array::{DType, RumpyArray};
use crate::array::dtype::{DTypeKind, TimeUnit, NAT, format_datetime64, parse_datetime64, convert_datetime64};

use super::{parse_dtype, PyRumpyArray};

/// Default business day calendar (Mon-Fri, no holidays).
const DEFAULT_WEEKMASK: [bool; 7] = [true, true, true, true, true, false, false];

/// Parse dates from Python input (string or list of strings).
fn parse_dates_input(dates: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(s) = dates.extract::<String>() {
        Ok(vec![parse_datetime64(&s, TimeUnit::Days)])
    } else if let Ok(list) = dates.downcast::<PyList>() {
        list.iter()
            .map(|item| {
                let s: String = item.extract()?;
                Ok(parse_datetime64(&s, TimeUnit::Days))
            })
            .collect()
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("dates must be string or list"))
    }
}

/// Test element-wise for NaT (not a time) and return result as a boolean array.
#[pyfunction]
pub fn isnat(arr: &PyRumpyArray) -> PyResult<PyRumpyArray> {
    let inner = &arr.inner;
    let dtype = inner.dtype();

    // Check if datetime64 or timedelta64
    if !matches!(dtype.kind(), DTypeKind::DateTime64(_) | DTypeKind::TimeDelta64(_)) {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "isnat is only valid for datetime or timedelta arrays"
        ));
    }

    let size = inner.size();
    let mut result = RumpyArray::zeros(inner.shape().to_vec(), DType::bool());
    let src_ptr = inner.data_ptr() as *const i64;
    let result_buf = std::sync::Arc::get_mut(result.buffer_mut()).unwrap();
    let dst_ptr = result_buf.as_mut_ptr() as *mut u8;

    for (i, offset) in inner.iter_offsets().enumerate() {
        let value = unsafe { *src_ptr.offset(offset as isize / 8) };
        unsafe { *dst_ptr.add(i) = if value == NAT { 1 } else { 0 }; }
    }

    Ok(PyRumpyArray::new(result))
}

/// Convert an array of datetime64 to an array of strings.
#[pyfunction]
#[pyo3(signature = (arr, unit=None))]
pub fn datetime_as_string(arr: &PyRumpyArray, unit: Option<&str>) -> PyResult<Vec<String>> {
    let inner = &arr.inner;

    let source_unit = match inner.dtype().kind() {
        DTypeKind::DateTime64(u) => u,
        _ => return Err(pyo3::exceptions::PyTypeError::new_err(
            "datetime_as_string requires datetime64 array"
        )),
    };

    let target_unit = match unit {
        Some(u) => TimeUnit::from_str(u).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unknown unit: {}", u))
        })?,
        None => source_unit,
    };

    let src_ptr = inner.data_ptr() as *const i64;
    let mut result = Vec::with_capacity(inner.size());

    for offset in inner.iter_offsets() {
        let value = unsafe { *src_ptr.offset(offset as isize / 8) };
        let target_value = if source_unit != target_unit {
            convert_datetime64(value, source_unit, target_unit)
        } else {
            value
        };
        result.push(if target_value == NAT {
            "NaT".to_string()
        } else {
            format_datetime64(target_value, target_unit)
        });
    }

    Ok(result)
}

/// Get information about datetime64 dtype.
/// Returns (unit_string, step_size) tuple.
#[pyfunction]
pub fn datetime_data(dtype_str: &str) -> PyResult<(String, i64)> {
    let dtype = parse_dtype(dtype_str)?;
    match dtype.kind() {
        DTypeKind::DateTime64(unit) | DTypeKind::TimeDelta64(unit) => {
            Ok((unit.as_str().to_string(), 1))
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "datetime_data requires datetime64 or timedelta64 dtype"
        )),
    }
}

// ============================================================================
// Business Day Calendar and Functions
// ============================================================================

/// Business day calendar class.
#[pyclass]
#[derive(Clone)]
pub struct BusDayCalendar {
    /// Boolean array: weekmask[0]=Mon, weekmask[6]=Sun. True=business day.
    pub weekmask: [bool; 7],
    /// List of holiday dates (as days since epoch).
    pub holidays: Vec<i64>,
}

#[pymethods]
impl BusDayCalendar {
    #[new]
    #[pyo3(signature = (weekmask=None, holidays=None))]
    fn new(weekmask: Option<&str>, holidays: Option<Vec<String>>) -> PyResult<Self> {
        // Default weekmask: Mon-Fri are business days
        let weekmask_arr = match weekmask {
            Some(mask) => parse_weekmask(mask)?,
            None => [true, true, true, true, true, false, false],
        };

        // Parse holidays
        let holidays_vec = match holidays {
            Some(h) => {
                h.iter()
                    .map(|s| parse_datetime64(s, TimeUnit::Days))
                    .collect()
            }
            None => Vec::new(),
        };

        Ok(BusDayCalendar {
            weekmask: weekmask_arr,
            holidays: holidays_vec,
        })
    }

    #[getter]
    fn weekmask(&self) -> Vec<bool> {
        self.weekmask.to_vec()
    }

    #[getter]
    fn holidays(&self) -> PyResult<PyRumpyArray> {
        let data: Vec<i64> = self.holidays.clone();
        Ok(PyRumpyArray::new(RumpyArray::from_vec_i64(data, DType::datetime64_d())))
    }
}

/// Parse weekmask string like "Mon Tue Wed Thu Fri" or "1111100".
fn parse_weekmask(mask: &str) -> PyResult<[bool; 7]> {
    let mask_lower = mask.to_lowercase();

    // Check for day name format
    if mask_lower.contains("mon") || mask_lower.contains("tue") {
        let mut result = [false; 7];
        if mask_lower.contains("mon") { result[0] = true; }
        if mask_lower.contains("tue") { result[1] = true; }
        if mask_lower.contains("wed") { result[2] = true; }
        if mask_lower.contains("thu") { result[3] = true; }
        if mask_lower.contains("fri") { result[4] = true; }
        if mask_lower.contains("sat") { result[5] = true; }
        if mask_lower.contains("sun") { result[6] = true; }
        return Ok(result);
    }

    // Check for binary format "1111100"
    let clean = mask.replace(" ", "");
    if clean.len() == 7 && clean.chars().all(|c| c == '0' || c == '1') {
        let mut result = [false; 7];
        for (i, c) in clean.chars().enumerate() {
            result[i] = c == '1';
        }
        return Ok(result);
    }

    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "Invalid weekmask: {}. Use day names (Mon Tue...) or binary (1111100)",
        mask
    )))
}

/// Check if dates are business days.
#[pyfunction]
#[pyo3(signature = (dates, busdaycal=None))]
pub fn is_busday(py: Python<'_>, dates: &Bound<'_, PyAny>, busdaycal: Option<&BusDayCalendar>) -> PyResult<PyObject> {
    let default_cal = BusDayCalendar { weekmask: DEFAULT_WEEKMASK, holidays: Vec::new() };
    let cal = busdaycal.unwrap_or(&default_cal);

    // Handle single string - return scalar bool
    if let Ok(s) = dates.extract::<String>() {
        let result = is_business_day(parse_datetime64(&s, TimeUnit::Days), cal);
        return Ok(result.into_pyobject(py)?.to_owned().into_any().unbind());
    }

    // Handle list of strings
    if let Ok(list) = dates.downcast::<PyList>() {
        let result: Vec<f64> = list.iter()
            .map(|item| -> PyResult<f64> {
                let s: String = item.extract()?;
                Ok(if is_business_day(parse_datetime64(&s, TimeUnit::Days), cal) { 1.0 } else { 0.0 })
            })
            .collect::<PyResult<_>>()?;
        return Ok(PyRumpyArray::new(RumpyArray::from_vec(result, DType::bool()))
            .into_pyobject(py)?.into_any().unbind());
    }

    // Handle rumpy array
    if let Ok(arr) = dates.extract::<PyRef<'_, PyRumpyArray>>() {
        let inner = &arr.inner;
        let mut result = RumpyArray::zeros(inner.shape().to_vec(), DType::bool());
        let src_ptr = inner.data_ptr() as *const i64;
        let dst_ptr = std::sync::Arc::get_mut(result.buffer_mut()).unwrap().as_mut_ptr() as *mut u8;

        for (i, offset) in inner.iter_offsets().enumerate() {
            let days = unsafe { *src_ptr.offset(offset as isize / 8) };
            unsafe { *dst_ptr.add(i) = if is_business_day(days, cal) { 1 } else { 0 }; }
        }
        return Ok(PyRumpyArray::new(result).into_pyobject(py)?.into_any().unbind());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "dates must be string, list of strings, or datetime64 array"
    ))
}

/// Check if a single day (days since epoch) is a business day.
fn is_business_day(days: i64, cal: &BusDayCalendar) -> bool {
    if days == NAT {
        return false;
    }

    // Check holidays
    if cal.holidays.contains(&days) {
        return false;
    }

    // Get day of week (0=Mon, 6=Sun)
    // 1970-01-01 was a Thursday (day 3)
    let dow = ((days % 7 + 7 + 3) % 7) as usize;
    cal.weekmask[dow]
}

/// Offset dates by business days.
#[pyfunction]
#[pyo3(signature = (dates, offsets, busdaycal=None))]
pub fn busday_offset(py: Python<'_>, dates: &Bound<'_, PyAny>, offsets: &Bound<'_, PyAny>, busdaycal: Option<&BusDayCalendar>) -> PyResult<PyObject> {
    let default_cal = BusDayCalendar { weekmask: DEFAULT_WEEKMASK, holidays: Vec::new() };
    let cal = busdaycal.unwrap_or(&default_cal);

    let dates_vec = parse_dates_input(dates)?;
    let is_scalar_date = dates.extract::<String>().is_ok();

    // Parse offsets
    let is_scalar_offset = offsets.extract::<i64>().is_ok();
    let offsets_vec: Vec<i64> = if let Ok(n) = offsets.extract::<i64>() {
        vec![n; dates_vec.len()]
    } else if let Ok(list) = offsets.downcast::<PyList>() {
        list.iter().map(|item| item.extract::<i64>()).collect::<PyResult<_>>()?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err("offsets must be int or list"));
    };

    let result: Vec<i64> = dates_vec.iter().zip(offsets_vec.iter())
        .map(|(&date, &offset)| busday_offset_single(date, offset, cal))
        .collect();

    // Return 0-d array for scalar inputs
    if is_scalar_date && is_scalar_offset {
        let arr = RumpyArray::from_vec_i64(result, DType::datetime64_d())
            .reshape(vec![]).expect("reshape to 0-d works for size 1");
        return Ok(PyRumpyArray::new(arr).into_pyobject(py)?.into_any().unbind());
    }

    Ok(PyRumpyArray::new(RumpyArray::from_vec_i64(result, DType::datetime64_d()))
        .into_pyobject(py)?.into_any().unbind())
}

/// Offset a single date by business days.
fn busday_offset_single(date: i64, offset: i64, cal: &BusDayCalendar) -> i64 {
    if date == NAT {
        return NAT;
    }

    let mut current = date;
    let mut remaining = offset.abs();
    let step = if offset >= 0 { 1 } else { -1 };

    // First, if current day is not a business day, find the next/prev business day
    // (NumPy raises an error by default, but we'll just move to nearest)

    while remaining > 0 {
        current += step;
        if is_business_day(current, cal) {
            remaining -= 1;
        }
    }

    current
}

/// Count business days between two dates.
#[pyfunction]
#[pyo3(signature = (begindates, enddates, busdaycal=None))]
pub fn busday_count(py: Python<'_>, begindates: &Bound<'_, PyAny>, enddates: &Bound<'_, PyAny>, busdaycal: Option<&BusDayCalendar>) -> PyResult<PyObject> {
    let default_cal = BusDayCalendar { weekmask: DEFAULT_WEEKMASK, holidays: Vec::new() };
    let cal = busdaycal.unwrap_or(&default_cal);

    let begin_vec = parse_dates_input(begindates)?;
    let end_vec = parse_dates_input(enddates)?;

    let result: Vec<i64> = begin_vec.iter().zip(end_vec.iter())
        .map(|(&begin, &end)| busday_count_single(begin, end, cal))
        .collect();

    // Return scalar for scalar inputs
    if begindates.extract::<String>().is_ok() && enddates.extract::<String>().is_ok() {
        return Ok(result[0].into_pyobject(py)?.into_any().unbind());
    }

    Ok(PyRumpyArray::new(RumpyArray::from_vec_i64(result, DType::int64()))
        .into_pyobject(py)?.into_any().unbind())
}

/// Count business days between two dates (exclusive of end date).
fn busday_count_single(begin: i64, end: i64, cal: &BusDayCalendar) -> i64 {
    if begin == NAT || end == NAT {
        return 0;
    }

    if begin >= end {
        return 0;
    }

    let mut count = 0i64;
    for day in begin..end {
        if is_business_day(day, cal) {
            count += 1;
        }
    }
    count
}

/// Factory function for busdaycalendar (lowercase for NumPy compatibility).
#[pyfunction]
#[pyo3(signature = (weekmask=None, holidays=None))]
pub fn busdaycalendar(weekmask: Option<&str>, holidays: Option<Vec<String>>) -> PyResult<BusDayCalendar> {
    BusDayCalendar::new(weekmask, holidays)
}

/// Register datetime functions with the module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BusDayCalendar>()?;
    m.add_function(wrap_pyfunction!(busdaycalendar, m)?)?;
    m.add_function(wrap_pyfunction!(isnat, m)?)?;
    m.add_function(wrap_pyfunction!(datetime_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(datetime_data, m)?)?;
    m.add_function(wrap_pyfunction!(is_busday, m)?)?;
    m.add_function(wrap_pyfunction!(busday_offset, m)?)?;
    m.add_function(wrap_pyfunction!(busday_count, m)?)?;
    Ok(())
}
