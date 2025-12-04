//! Gufunc signature parsing.
//!
//! Parses NumPy-style gufunc signatures like "(m,n),(n,p)->(m,p)".

/// A named dimension in a signature (e.g., "m", "n").
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DimName(pub String);

impl DimName {
    pub fn new(name: &str) -> Self {
        DimName(name.to_string())
    }
}

/// Core dimensions for one input or output.
/// Examples:
///   - `(m,n)` -> CoreDims with dims ["m", "n"]
///   - `()` -> CoreDims with empty dims (scalar)
#[derive(Clone, Debug, PartialEq)]
pub struct CoreDims {
    pub dims: Vec<DimName>,
}

impl CoreDims {
    pub fn new(dims: Vec<DimName>) -> Self {
        CoreDims { dims }
    }

    pub fn scalar() -> Self {
        CoreDims { dims: vec![] }
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Parse from string like "(m,n)" or "()" or "(n)".
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if !s.starts_with('(') || !s.ends_with(')') {
            return None;
        }

        let inner = &s[1..s.len() - 1].trim();
        if inner.is_empty() {
            return Some(CoreDims::scalar());
        }

        let dims: Vec<DimName> = inner
            .split(',')
            .map(|d| DimName::new(d.trim()))
            .collect();

        // Validate dimension names are non-empty
        if dims.iter().any(|d| d.0.is_empty()) {
            return None;
        }

        Some(CoreDims::new(dims))
    }
}

/// Full gufunc signature.
/// Example: "(m,n),(n,p)->(m,p)" for matmul.
#[derive(Clone, Debug, PartialEq)]
pub struct GufuncSignature {
    pub inputs: Vec<CoreDims>,
    pub outputs: Vec<CoreDims>,
}

impl GufuncSignature {
    pub fn new(inputs: Vec<CoreDims>, outputs: Vec<CoreDims>) -> Self {
        GufuncSignature { inputs, outputs }
    }

    /// Parse signature string like "(m,n),(n,p)->(m,p)".
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();

        // Split on "->"
        let arrow_pos = s.find("->")?;
        let inputs_str = &s[..arrow_pos];
        let outputs_str = &s[arrow_pos + 2..];

        let inputs = parse_core_dims_list(inputs_str)?;
        let outputs = parse_core_dims_list(outputs_str)?;

        if inputs.is_empty() {
            return None;
        }

        Some(GufuncSignature { inputs, outputs })
    }

    /// Number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Core ndim for input i.
    pub fn input_core_ndim(&self, i: usize) -> usize {
        self.inputs[i].ndim()
    }

    /// Core ndim for output i.
    pub fn output_core_ndim(&self, i: usize) -> usize {
        self.outputs[i].ndim()
    }
}

/// Parse a comma-separated list of CoreDims like "(m,n),(n,p)".
fn parse_core_dims_list(s: &str) -> Option<Vec<CoreDims>> {
    let s = s.trim();
    if s.is_empty() {
        return Some(vec![]);
    }

    let mut result = vec![];
    let mut depth = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let segment = &s[start..=i];
                    result.push(CoreDims::parse(segment)?);
                }
            }
            ',' if depth == 0 => {
                start = i + 1;
            }
            _ => {}
        }
    }

    // Check balanced parentheses
    if depth != 0 {
        return None;
    }

    Some(result)
}
