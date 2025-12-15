//! DType system extension functions for NumPy parity.
//!
//! Provides finfo, iinfo, promote_types, result_type, can_cast,
//! common_type, issubdtype, isdtype.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use crate::array::DType;
use crate::array::dtype::DTypeKind;

/// Floating-point type information (matches numpy.finfo).
#[pyclass(name = "finfo")]
pub struct PyFinfo {
    #[pyo3(get)]
    pub dtype: String,
    #[pyo3(get)]
    pub bits: u8,
    #[pyo3(get)]
    pub eps: f64,
    #[pyo3(get)]
    pub max: f64,
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub tiny: f64,
    #[pyo3(get)]
    pub smallest_normal: f64,
    #[pyo3(get)]
    pub smallest_subnormal: f64,
    #[pyo3(get)]
    pub machep: i32,
    #[pyo3(get)]
    pub negep: i32,
    #[pyo3(get)]
    pub minexp: i32,
    #[pyo3(get)]
    pub maxexp: i32,
    #[pyo3(get)]
    pub precision: u8,
    #[pyo3(get)]
    pub resolution: f64,
}

#[pymethods]
impl PyFinfo {
    #[new]
    fn new(dtype_arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dtype_str = parse_dtype_arg(dtype_arg)?;

        match dtype_str.as_str() {
            "float16" => Ok(Self {
                dtype: "float16".to_string(),
                bits: 16,
                eps: 0.000977,  // 2^-10
                max: 65504.0,
                min: -65504.0,
                tiny: 6.104e-5,  // smallest positive normal
                smallest_normal: 6.104e-5,
                smallest_subnormal: 5.96e-8,
                machep: -10,
                negep: -11,
                minexp: -14,
                maxexp: 16,
                precision: 3,
                resolution: 1e-3,
            }),
            "float32" => Ok(Self {
                dtype: "float32".to_string(),
                bits: 32,
                eps: f32::EPSILON as f64,
                max: f32::MAX as f64,
                min: f32::MIN as f64,
                tiny: f32::MIN_POSITIVE as f64,
                smallest_normal: f32::MIN_POSITIVE as f64,
                smallest_subnormal: 1.4e-45,
                machep: -23,
                negep: -24,
                minexp: -126,
                maxexp: 128,
                precision: 6,
                resolution: 1e-6,
            }),
            "float64" | "float" => Ok(Self {
                dtype: "float64".to_string(),
                bits: 64,
                eps: f64::EPSILON,
                max: f64::MAX,
                min: f64::MIN,
                tiny: f64::MIN_POSITIVE,
                smallest_normal: f64::MIN_POSITIVE,
                smallest_subnormal: 5e-324,
                machep: -52,
                negep: -53,
                minexp: -1022,
                maxexp: 1024,
                precision: 15,
                resolution: 1e-15,
            }),
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("finfo() dtype must be a floating point type, not '{}'", dtype_str)
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "finfo(resolution={}, min={:e}, max={:e}, dtype={})",
            self.resolution, self.min, self.max, self.dtype
        )
    }
}

/// Integer type information (matches numpy.iinfo).
#[pyclass(name = "iinfo")]
pub struct PyIinfo {
    #[pyo3(get)]
    pub dtype: String,
    #[pyo3(get)]
    pub bits: u8,
    #[pyo3(get)]
    pub min: i128,
    #[pyo3(get)]
    pub max: u128,
}

#[pymethods]
impl PyIinfo {
    #[new]
    fn new(dtype_arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dtype_str = parse_dtype_arg(dtype_arg)?;

        match dtype_str.as_str() {
            "int8" => Ok(Self {
                dtype: "int8".to_string(),
                bits: 8,
                min: i8::MIN as i128,
                max: i8::MAX as u128,
            }),
            "int16" => Ok(Self {
                dtype: "int16".to_string(),
                bits: 16,
                min: i16::MIN as i128,
                max: i16::MAX as u128,
            }),
            "int32" => Ok(Self {
                dtype: "int32".to_string(),
                bits: 32,
                min: i32::MIN as i128,
                max: i32::MAX as u128,
            }),
            "int64" | "int" => Ok(Self {
                dtype: "int64".to_string(),
                bits: 64,
                min: i64::MIN as i128,
                max: i64::MAX as u128,
            }),
            "uint8" => Ok(Self {
                dtype: "uint8".to_string(),
                bits: 8,
                min: 0,
                max: u8::MAX as u128,
            }),
            "uint16" => Ok(Self {
                dtype: "uint16".to_string(),
                bits: 16,
                min: 0,
                max: u16::MAX as u128,
            }),
            "uint32" => Ok(Self {
                dtype: "uint32".to_string(),
                bits: 32,
                min: 0,
                max: u32::MAX as u128,
            }),
            "uint64" => Ok(Self {
                dtype: "uint64".to_string(),
                bits: 64,
                min: 0,
                max: u64::MAX as u128,
            }),
            "bool" => Ok(Self {
                dtype: "bool".to_string(),
                bits: 8,
                min: 0,
                max: 1,
            }),
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("iinfo() dtype must be an integer type, not '{}'", dtype_str)
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "iinfo(min={}, max={}, dtype={})",
            self.min, self.max, self.dtype
        )
    }
}

/// Parse dtype argument which can be string, numpy dtype, or our dtype.
fn parse_dtype_arg(arg: &Bound<'_, PyAny>) -> PyResult<String> {
    // Try string first
    if let Ok(s) = arg.extract::<String>() {
        return Ok(normalize_dtype_string(&s));
    }

    // Try extracting from numpy dtype via name attribute
    if let Ok(name) = arg.getattr("name") {
        if let Ok(s) = name.extract::<String>() {
            return Ok(normalize_dtype_string(&s));
        }
    }

    // Try extracting from numpy dtype via str()
    if let Ok(s) = arg.str() {
        return Ok(normalize_dtype_string(&s.to_string()));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "dtype argument must be a dtype specifier"
    ))
}

/// Normalize dtype string to canonical form.
fn normalize_dtype_string(s: &str) -> String {
    match s {
        "f2" | "<f2" | "float16" => "float16".to_string(),
        "f4" | "<f4" | "float32" => "float32".to_string(),
        "f8" | "<f8" | "float64" | "float" => "float64".to_string(),
        "i1" | "|i1" | "int8" => "int8".to_string(),
        "i2" | "<i2" | "int16" => "int16".to_string(),
        "i4" | "<i4" | "int32" => "int32".to_string(),
        "i8" | "<i8" | "int64" | "int" => "int64".to_string(),
        "u1" | "|u1" | "uint8" => "uint8".to_string(),
        "u2" | "<u2" | "uint16" => "uint16".to_string(),
        "u4" | "<u4" | "uint32" => "uint32".to_string(),
        "u8" | "<u8" | "uint64" => "uint64".to_string(),
        "?" | "|b1" | "bool" => "bool".to_string(),
        "c8" | "<c8" | "complex64" => "complex64".to_string(),
        "c16" | "<c16" | "complex128" => "complex128".to_string(),
        _ => s.to_string(),
    }
}

/// Convert dtype string to DType.
fn dtype_from_string(s: &str) -> Option<DType> {
    let normalized = normalize_dtype_string(s);
    DType::parse(&normalized)
}

/// Find common dtype for two dtype specifications.
#[pyfunction]
pub fn promote_types(dtype1: &Bound<'_, PyAny>, dtype2: &Bound<'_, PyAny>) -> PyResult<String> {
    let s1 = parse_dtype_arg(dtype1)?;
    let s2 = parse_dtype_arg(dtype2)?;

    let d1 = dtype_from_string(&s1).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unknown dtype: {}", s1))
    })?;
    let d2 = dtype_from_string(&s2).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unknown dtype: {}", s2))
    })?;

    let result = crate::array::dtype::promote_dtype(&d1, &d2);
    Ok(result.ops().name().to_string())
}

/// Determine result dtype for a set of operands.
/// Accepts dtypes, arrays, or scalars.
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn result_type(args: &Bound<'_, PyTuple>) -> PyResult<String> {
    if args.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "result_type() requires at least one argument"
        ));
    }

    let mut result_dtype: Option<DType> = None;

    for arg in args.iter() {
        let dtype = extract_dtype_from_operand(&arg)?;
        result_dtype = Some(match result_dtype {
            None => dtype,
            Some(current) => crate::array::dtype::promote_dtype(&current, &dtype),
        });
    }

    Ok(result_dtype.unwrap().ops().name().to_string())
}

/// Extract dtype from an operand (array, scalar, or dtype).
fn extract_dtype_from_operand(arg: &Bound<'_, PyAny>) -> PyResult<DType> {
    // Check if it has a dtype attribute (numpy array, our array, or dtype)
    if let Ok(dtype_attr) = arg.getattr("dtype") {
        let dtype_str = parse_dtype_arg(&dtype_attr)?;
        if let Some(dt) = dtype_from_string(&dtype_str) {
            return Ok(dt);
        }
    }

    // Check if it's a dtype string
    if let Ok(s) = arg.extract::<String>() {
        if let Some(dt) = dtype_from_string(&s) {
            return Ok(dt);
        }
    }

    // Check for Python scalars - order matters (bool before int)
    if arg.extract::<bool>().is_ok() {
        return Ok(DType::bool());
    }
    if arg.extract::<i64>().is_ok() {
        return Ok(DType::int64());
    }
    if arg.extract::<f64>().is_ok() {
        return Ok(DType::float64());
    }

    // Try complex via Python's complex type
    if arg.is_instance_of::<pyo3::types::PyComplex>() {
        return Ok(DType::complex128());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        format!("Cannot determine dtype from argument: {:?}", arg)
    ))
}

/// Check if a cast between dtypes is possible.
#[pyfunction]
#[pyo3(signature = (from_dtype, to_dtype, casting="safe"))]
pub fn can_cast(from_dtype: &Bound<'_, PyAny>, to_dtype: &Bound<'_, PyAny>, casting: &str) -> PyResult<bool> {
    let from_str = parse_dtype_arg(from_dtype)?;
    let to_str = parse_dtype_arg(to_dtype)?;

    let from_dt = dtype_from_string(&from_str).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unknown dtype: {}", from_str))
    })?;
    let to_dt = dtype_from_string(&to_str).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unknown dtype: {}", to_str))
    })?;

    match casting {
        "no" => Ok(from_dt == to_dt),
        "equiv" => Ok(from_dt == to_dt || (from_dt.itemsize() == to_dt.itemsize())),
        "safe" => Ok(can_cast_safe(&from_dt, &to_dt)),
        "same_kind" => Ok(can_cast_same_kind(&from_dt, &to_dt)),
        "unsafe" => Ok(true),  // Anything can cast unsafely
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe', not '{}'", casting)
        )),
    }
}

/// Check if cast is safe (no precision/range loss).
fn can_cast_safe(from: &DType, to: &DType) -> bool {
    use DTypeKind::*;

    if from == to {
        return true;
    }

    let fk = from.kind();
    let tk = to.kind();

    // Bool can safely cast to any numeric
    if matches!(fk, Bool) {
        return matches!(tk, Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64 |
                           Float16 | Float32 | Float64 | Complex64 | Complex128);
    }

    // Integers to larger integers or floats
    let int_bits = |k: &DTypeKind| match k {
        Int8 | Uint8 => 8,
        Int16 | Uint16 => 16,
        Int32 | Uint32 => 32,
        Int64 | Uint64 => 64,
        _ => 0,
    };

    let is_signed = |k: &DTypeKind| matches!(k, Int8 | Int16 | Int32 | Int64);
    let is_unsigned = |k: &DTypeKind| matches!(k, Uint8 | Uint16 | Uint32 | Uint64);
    let is_float = |k: &DTypeKind| matches!(k, Float16 | Float32 | Float64);
    let is_complex = |k: &DTypeKind| matches!(k, Complex64 | Complex128);

    // Float to larger float
    if is_float(&fk) && is_float(&tk) {
        return from.itemsize() <= to.itemsize();
    }

    // Float to complex
    if is_float(&fk) && is_complex(&tk) {
        return matches!(
            (&fk, &tk),
            (Float32, Complex64) | (Float32, Complex128) | (Float64, Complex128) | (Float16, _)
        );
    }

    // Complex to larger complex
    if is_complex(&fk) && is_complex(&tk) {
        return from.itemsize() <= to.itemsize();
    }

    // Integer to float (with precision considerations)
    if (is_signed(&fk) || is_unsigned(&fk)) && is_float(&tk) {
        let from_bits = int_bits(&fk);
        return match tk {
            Float64 => from_bits <= 32,  // float64 can exactly represent int32
            Float32 => from_bits <= 16,  // float32 can exactly represent int16
            Float16 => from_bits <= 8,   // float16 can exactly represent small ints
            _ => false,
        };
    }

    // Integer to complex
    if (is_signed(&fk) || is_unsigned(&fk)) && is_complex(&tk) {
        return true;  // Complex can hold any integer
    }

    // Signed int to larger signed int
    if is_signed(&fk) && is_signed(&tk) {
        return int_bits(&fk) <= int_bits(&tk);
    }

    // Unsigned to larger unsigned
    if is_unsigned(&fk) && is_unsigned(&tk) {
        return int_bits(&fk) <= int_bits(&tk);
    }

    // Unsigned to signed (needs extra bit)
    if is_unsigned(&fk) && is_signed(&tk) {
        return int_bits(&fk) < int_bits(&tk);
    }

    false
}

/// Check if cast is same_kind (numbers to numbers of same kind).
fn can_cast_same_kind(from: &DType, to: &DType) -> bool {
    use DTypeKind::*;

    if can_cast_safe(from, to) {
        return true;
    }

    let fk = from.kind();
    let tk = to.kind();

    let is_int = |k: &DTypeKind| matches!(k, Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64);
    let is_float = |k: &DTypeKind| matches!(k, Float16 | Float32 | Float64);
    let is_complex = |k: &DTypeKind| matches!(k, Complex64 | Complex128);

    // Int to int (any direction)
    if is_int(&fk) && is_int(&tk) {
        return true;
    }

    // Float to float (any direction)
    if is_float(&fk) && is_float(&tk) {
        return true;
    }

    // Complex to complex (any direction)
    if is_complex(&fk) && is_complex(&tk) {
        return true;
    }

    false
}

/// Find common type for array operands (returns scalar type for computations).
/// Always returns at least float64 for integers.
#[pyfunction]
#[pyo3(signature = (*arrays))]
pub fn common_type(arrays: &Bound<'_, PyTuple>) -> PyResult<PyObject> {
    let py = arrays.py();

    if arrays.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "common_type() requires at least one argument"
        ));
    }

    let mut has_complex = false;
    let mut max_float_size = 0usize;

    for arr in arrays.iter() {
        let dtype = extract_dtype_from_operand(&arr)?;
        match dtype.kind() {
            DTypeKind::Complex128 => {
                has_complex = true;
                max_float_size = max_float_size.max(16);
            }
            DTypeKind::Complex64 => {
                has_complex = true;
                max_float_size = max_float_size.max(8);
            }
            DTypeKind::Float64 => {
                max_float_size = max_float_size.max(8);
            }
            DTypeKind::Float32 => {
                max_float_size = max_float_size.max(4);
            }
            DTypeKind::Float16 => {
                max_float_size = max_float_size.max(2);
            }
            _ => {
                // Integers promote to float64
                max_float_size = max_float_size.max(8);
            }
        }
    }

    // Return the appropriate numpy dtype type
    let type_name = if has_complex {
        if max_float_size >= 16 { "complex128" } else { "complex64" }
    } else if max_float_size >= 8 {
        "float64"
    } else if max_float_size >= 4 {
        "float32"
    } else {
        "float64"  // Default to float64
    };

    // Return the type object (like numpy does)
    let numpy = py.import("numpy")?;
    let dtype_type = numpy.getattr(type_name)?;
    Ok(dtype_type.unbind())
}

/// Check if dtype is a subtype of a type category.
#[pyfunction]
pub fn issubdtype(dtype: &Bound<'_, PyAny>, dtype_or_category: &Bound<'_, PyAny>) -> PyResult<bool> {
    let dtype_str = parse_dtype_arg(dtype)?;
    let dt = dtype_from_string(&dtype_str).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unknown dtype: {}", dtype_str))
    })?;

    // Check if second arg is a category (numpy generic type)
    if let Ok(name) = dtype_or_category.getattr("__name__") {
        if let Ok(s) = name.extract::<String>() {
            return Ok(check_dtype_category(&dt, &s));
        }
    }

    // Try as a string category
    if let Ok(s) = dtype_or_category.extract::<String>() {
        return Ok(check_dtype_category(&dt, &s));
    }

    // Check if it's a specific dtype
    let other_str = parse_dtype_arg(dtype_or_category)?;
    let other_dt = dtype_from_string(&other_str).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unknown dtype: {}", other_str))
    })?;

    Ok(dt == other_dt)
}

/// Check if dtype belongs to a category.
fn check_dtype_category(dt: &DType, category: &str) -> bool {
    use DTypeKind::*;
    let kind = dt.kind();

    match category {
        "integer" | "signedinteger" | "int_" => matches!(kind, Int8 | Int16 | Int32 | Int64),
        "unsignedinteger" | "uint" => matches!(kind, Uint8 | Uint16 | Uint32 | Uint64),
        "floating" | "float_" => matches!(kind, Float16 | Float32 | Float64),
        "complexfloating" | "complex_" => matches!(kind, Complex64 | Complex128),
        "number" => matches!(kind, Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64 |
                                   Float16 | Float32 | Float64 | Complex64 | Complex128),
        "bool_" | "bool8" => matches!(kind, Bool),
        "generic" => true,  // Everything is a subtype of generic
        _ => false,
    }
}

/// Check if dtype matches a specified kind (NumPy 2.0+ style).
#[pyfunction]
pub fn isdtype(dtype: &Bound<'_, PyAny>, kind: &Bound<'_, PyAny>) -> PyResult<bool> {
    let dtype_str = parse_dtype_arg(dtype)?;
    let dt = dtype_from_string(&dtype_str).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unknown dtype: {}", dtype_str))
    })?;

    // Handle tuple of kinds
    if let Ok(tuple) = kind.downcast::<PyTuple>() {
        for item in tuple.iter() {
            if check_isdtype_kind(&dt, &item)? {
                return Ok(true);
            }
        }
        return Ok(false);
    }

    check_isdtype_kind(&dt, kind)
}

/// Check single isdtype kind.
fn check_isdtype_kind(dt: &DType, kind: &Bound<'_, PyAny>) -> PyResult<bool> {
    use DTypeKind::*;
    let dtype_kind = dt.kind();

    // Handle string kind specifiers
    if let Ok(s) = kind.extract::<String>() {
        return Ok(match s.as_str() {
            "bool" => matches!(dtype_kind, Bool),
            "signed integer" => matches!(dtype_kind, Int8 | Int16 | Int32 | Int64),
            "unsigned integer" => matches!(dtype_kind, Uint8 | Uint16 | Uint32 | Uint64),
            "integral" => matches!(dtype_kind, Bool | Int8 | Int16 | Int32 | Int64 |
                                              Uint8 | Uint16 | Uint32 | Uint64),
            "real floating" => matches!(dtype_kind, Float16 | Float32 | Float64),
            "complex floating" => matches!(dtype_kind, Complex64 | Complex128),
            "numeric" => matches!(dtype_kind, Int8 | Int16 | Int32 | Int64 |
                                             Uint8 | Uint16 | Uint32 | Uint64 |
                                             Float16 | Float32 | Float64 |
                                             Complex64 | Complex128),
            _ => false,
        });
    }

    // Handle numpy dtype as kind
    let other_str = parse_dtype_arg(kind)?;
    if let Some(other_dt) = dtype_from_string(&other_str) {
        return Ok(*dt == other_dt);
    }

    Ok(false)
}

/// Register dtype functions with the module.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyFinfo>()?;
    parent.add_class::<PyIinfo>()?;
    parent.add_function(wrap_pyfunction!(promote_types, parent)?)?;
    parent.add_function(wrap_pyfunction!(result_type, parent)?)?;
    parent.add_function(wrap_pyfunction!(can_cast, parent)?)?;
    parent.add_function(wrap_pyfunction!(common_type, parent)?)?;
    parent.add_function(wrap_pyfunction!(issubdtype, parent)?)?;
    parent.add_function(wrap_pyfunction!(isdtype, parent)?)?;
    Ok(())
}
