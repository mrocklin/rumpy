/// Data types supported by rumpy arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
}

impl DType {
    /// Size in bytes of one element.
    pub const fn itemsize(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Bool => 1,
        }
    }

    /// NumPy type string for __array_interface__.
    pub fn typestr(&self) -> &'static str {
        match self {
            DType::Float32 => "<f4",
            DType::Float64 => "<f8",
            DType::Int32 => "<i4",
            DType::Int64 => "<i8",
            DType::Bool => "|b1",
        }
    }

    /// Type character for buffer protocol format.
    pub fn format_char(&self) -> &'static str {
        match self {
            DType::Float32 => "f",
            DType::Float64 => "d",
            DType::Int32 => "i",
            DType::Int64 => "q",
            DType::Bool => "?",
        }
    }

    /// Parse dtype from string (numpy-style).
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "float32" | "f4" | "<f4" => Some(DType::Float32),
            "float64" | "f8" | "<f8" | "float" => Some(DType::Float64),
            "int32" | "i4" | "<i4" => Some(DType::Int32),
            "int64" | "i8" | "<i8" | "int" => Some(DType::Int64),
            "bool" | "?" | "|b1" => Some(DType::Bool),
            _ => None,
        }
    }
}
