//! Einstein summation implementation.
//!
//! Supports NumPy-compatible einsum notation for tensor contractions.
//! Reference: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
//!
//! Performance optimizations:
//! - Binary contractions dispatch to BLAS matmul via reshape/transpose
//! - Common patterns (transpose, trace, diagonal, sum, outer) use fast paths
//! - Views returned where possible (diagonal, transpose)

use crate::array::{promote_dtype, DType, RumpyArray};
use std::collections::{HashMap, HashSet};

/// Compute output dtype for einsum based on input dtypes (NumPy-compatible promotion).
fn compute_einsum_dtype(operands: &[&RumpyArray]) -> DType {
    if operands.is_empty() {
        return DType::float64();
    }
    let mut dtype = operands[0].dtype().clone();
    for op in &operands[1..] {
        dtype = promote_dtype(&dtype, &op.dtype());
    }
    dtype
}

/// Parsed einsum subscript.
#[derive(Debug, Clone)]
pub struct ParsedSubscript {
    /// Input subscripts, one per operand (e.g., ["ij", "jk"])
    pub inputs: Vec<Vec<char>>,
    /// Output subscript (e.g., ['i', 'k'])
    pub output: Vec<char>,
    /// Whether output was explicit (had ->)
    pub explicit_output: bool,
}

/// Parse einsum subscript string.
///
/// Handles:
/// - Explicit: "ij,jk->ik"
/// - Implicit: "ij,jk" (output is alphabetical union of non-repeated indices)
pub fn parse_subscript(subscript: &str) -> Result<ParsedSubscript, String> {
    // Remove whitespace
    let subscript: String = subscript.chars().filter(|c| !c.is_whitespace()).collect();

    let (input_part, output_part, explicit_output) = if subscript.contains("->") {
        let parts: Vec<&str> = subscript.split("->").collect();
        if parts.len() != 2 {
            return Err("Invalid subscript: multiple -> found".to_string());
        }
        (parts[0], Some(parts[1]), true)
    } else {
        (subscript.as_str(), None, false)
    };

    // Parse input subscripts
    let input_strs: Vec<&str> = input_part.split(',').collect();
    let inputs: Vec<Vec<char>> = input_strs
        .iter()
        .map(|s| s.chars().collect())
        .collect();

    // Validate subscripts (only lowercase letters allowed for now)
    for (i, sub) in inputs.iter().enumerate() {
        for c in sub {
            if !c.is_ascii_lowercase() {
                return Err(format!("Invalid subscript character '{}' in operand {}", c, i));
            }
        }
    }

    // Determine output subscripts
    let output = if let Some(out_str) = output_part {
        let out_chars: Vec<char> = out_str.chars().collect();
        for c in &out_chars {
            if !c.is_ascii_lowercase() && *c != '.' {
                return Err(format!("Invalid output subscript character '{}'", c));
            }
        }
        out_chars
    } else {
        // Implicit output: union of indices that appear exactly once across all inputs
        compute_implicit_output(&inputs)
    };

    Ok(ParsedSubscript {
        inputs,
        output,
        explicit_output,
    })
}

/// Compute implicit output indices.
///
/// According to NumPy rules:
/// - Indices appearing once in the full expression appear in output (alphabetical order)
/// - Indices appearing multiple times are summed over
fn compute_implicit_output(inputs: &[Vec<char>]) -> Vec<char> {
    let mut counts: HashMap<char, usize> = HashMap::new();
    for sub in inputs {
        for &c in sub {
            *counts.entry(c).or_insert(0) += 1;
        }
    }

    // Indices appearing exactly once, sorted
    let mut output: Vec<char> = counts
        .iter()
        .filter(|(_, &count)| count == 1)
        .map(|(&c, _)| c)
        .collect();
    output.sort();
    output
}

/// Validate parsed subscript against operand shapes.
pub fn validate_subscript(
    parsed: &ParsedSubscript,
    shapes: &[&[usize]],
) -> Result<HashMap<char, usize>, String> {
    if parsed.inputs.len() != shapes.len() {
        return Err(format!(
            "Number of subscripts ({}) doesn't match number of operands ({})",
            parsed.inputs.len(),
            shapes.len()
        ));
    }

    // Build dimension mapping
    let mut dim_map: HashMap<char, usize> = HashMap::new();

    for (i, (sub, shape)) in parsed.inputs.iter().zip(shapes.iter()).enumerate() {
        if sub.len() != shape.len() {
            return Err(format!(
                "Operand {} has {} dimensions but subscript '{}' has {} indices",
                i,
                shape.len(),
                sub.iter().collect::<String>(),
                sub.len()
            ));
        }

        for (j, &c) in sub.iter().enumerate() {
            if let Some(&existing_dim) = dim_map.get(&c) {
                if existing_dim != shape[j] {
                    return Err(format!(
                        "Dimension mismatch for index '{}': {} vs {}",
                        c, existing_dim, shape[j]
                    ));
                }
            } else {
                dim_map.insert(c, shape[j]);
            }
        }
    }

    // Validate output indices exist in inputs
    for &c in &parsed.output {
        if !dim_map.contains_key(&c) {
            return Err(format!(
                "Output index '{}' not found in any input subscript",
                c
            ));
        }
    }

    Ok(dim_map)
}

/// Compute output shape from parsed subscript and dimension map.
pub fn compute_output_shape(parsed: &ParsedSubscript, dim_map: &HashMap<char, usize>) -> Vec<usize> {
    parsed.output.iter().map(|c| dim_map[c]).collect()
}

/// Core einsum execution.
///
/// This uses a general n-dimensional iteration approach.
pub fn einsum(
    subscript: &str,
    operands: &[&RumpyArray],
) -> Result<RumpyArray, String> {
    let parsed = parse_subscript(subscript)?;
    let shapes: Vec<&[usize]> = operands.iter().map(|a| a.shape()).collect();
    let dim_map = validate_subscript(&parsed, &shapes)?;
    let output_shape = compute_output_shape(&parsed, &dim_map);

    // Try optimized paths for common patterns (these preserve dtype properly)
    if let Some(result) = try_optimized_path(&parsed, operands, &dim_map) {
        return Ok(result);
    }

    // Compute output dtype using NumPy-compatible promotion rules
    let out_dtype = compute_einsum_dtype(operands);

    // Create output array
    let out_size: usize = output_shape.iter().product();
    let result = if output_shape.is_empty() {
        // Scalar result
        RumpyArray::zeros(vec![1], out_dtype.clone())
    } else {
        RumpyArray::zeros(output_shape.clone(), out_dtype.clone())
    };

    // Handle empty result
    if out_size == 0 {
        return Ok(result);
    }

    // Find contraction indices (appear in inputs but not output)
    let output_set: HashSet<char> = parsed.output.iter().copied().collect();
    let mut contraction_indices: Vec<char> = dim_map
        .keys()
        .filter(|c| !output_set.contains(c))
        .copied()
        .collect();
    contraction_indices.sort(); // For deterministic order

    // Compute strides for each index in output
    let output_strides = compute_strides(&output_shape);

    // Execute based on number of operands and complexity
    match operands.len() {
        1 => einsum_single(
            &parsed,
            operands[0],
            &result,
            &dim_map,
            &contraction_indices,
        ),
        2 => einsum_binary(
            &parsed,
            operands[0],
            operands[1],
            &result,
            &dim_map,
            &contraction_indices,
            &output_strides,
        ),
        _ => einsum_multi(
            &parsed,
            operands,
            &result,
            &dim_map,
            &contraction_indices,
            &output_strides,
        ),
    }?;

    // Handle scalar output
    if output_shape.is_empty() {
        result.reshape(vec![]).ok_or_else(|| "Failed to reshape to scalar".to_string())
    } else {
        Ok(result)
    }
}

/// Try optimized paths for common einsum patterns.
fn try_optimized_path(
    parsed: &ParsedSubscript,
    operands: &[&RumpyArray],
    _dim_map: &HashMap<char, usize>,
) -> Option<RumpyArray> {
    match operands.len() {
        1 => try_single_operand_optimized(parsed, operands[0]),
        2 => try_binary_optimized(parsed, operands[0], operands[1]),
        _ => None,
    }
}

/// Optimized paths for single-operand einsum.
fn try_single_operand_optimized(parsed: &ParsedSubscript, a: &RumpyArray) -> Option<RumpyArray> {
    let in_sub = &parsed.inputs[0];
    let out_sub = &parsed.output;

    // Check for transpose: output is permutation of input (no repeated indices)
    let in_set: HashSet<char> = in_sub.iter().copied().collect();
    let out_set: HashSet<char> = out_sub.iter().copied().collect();

    if in_set.len() == in_sub.len() && out_set.len() == out_sub.len() && in_set == out_set {
        // Pure transpose - build permutation and use transpose_axes (preserves dtype)
        let perm: Vec<usize> = out_sub.iter()
            .map(|c| in_sub.iter().position(|x| x == c).unwrap())
            .collect();
        return Some(a.transpose_axes(&perm));
    }

    // Check for trace: ii-> (repeated index, scalar output)
    if in_sub.len() == 2 && in_sub[0] == in_sub[1] && out_sub.is_empty() {
        // trace() returns f64, create scalar with input dtype
        let trace_val = a.trace();
        return Some(RumpyArray::full(vec![], trace_val, a.dtype().clone()));
    }

    // Check for diagonal: ii->i (repeated index, vector output) - returns view with same dtype
    if in_sub.len() == 2 && in_sub[0] == in_sub[1] && out_sub.len() == 1 && out_sub[0] == in_sub[0] {
        return Some(a.diagonal());
    }

    // Check for sum over axis: ij->i or ij->j (sum_axis preserves dtype)
    if in_sub.len() == 2 && in_set.len() == 2 && out_sub.len() == 1 {
        let sum_axis = if out_sub[0] == in_sub[0] { 1 } else { 0 };
        return Some(a.sum_axis(sum_axis));
    }

    // Check for total sum: ij-> (sum all elements) - sum() returns f64, use input dtype
    if !in_sub.is_empty() && in_set.len() == in_sub.len() && out_sub.is_empty() {
        let sum_val = a.sum();
        return Some(RumpyArray::full(vec![], sum_val, a.dtype().clone()));
    }

    None
}

/// Optimized paths for binary einsum.
fn try_binary_optimized(parsed: &ParsedSubscript, a: &RumpyArray, b: &RumpyArray) -> Option<RumpyArray> {
    let a_sub = &parsed.inputs[0];
    let b_sub = &parsed.inputs[1];
    let out_sub = &parsed.output;

    // Check for outer product: i,j->ij (no shared indices)
    let a_set: HashSet<char> = a_sub.iter().copied().collect();
    let b_set: HashSet<char> = b_sub.iter().copied().collect();

    if a_set.is_disjoint(&b_set) {
        // Outer product (1D x 1D -> 2D) - outer() uses dtype promotion
        if a.ndim() == 1 && b.ndim() == 1 && out_sub.len() == 2 {
            let outer_result = crate::ops::outer::outer(a, b)?;
            let expected_out: Vec<char> = a_sub.iter().chain(b_sub.iter()).copied().collect();
            return Some(if expected_out == *out_sub {
                outer_result
            } else {
                outer_result.transpose()
            });
        }
    }

    // Check for hadamard: i,i->i or ij,ij->ij (element-wise multiply) - binary_op uses promotion
    if a_sub == b_sub && a_sub == out_sub {
        return a.binary_op(b, crate::ops::BinaryOp::Mul).ok();
    }

    None
}

/// Compute C-contiguous strides for a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Single-operand einsum (trace, diagonal, transpose, sum, etc.)
fn einsum_single(
    parsed: &ParsedSubscript,
    a: &RumpyArray,
    result: &RumpyArray,
    dim_map: &HashMap<char, usize>,
    contraction_indices: &[char],
) -> Result<(), String> {
    let a_sub = &parsed.inputs[0];
    let out_sub = &parsed.output;

    // Find which output index corresponds to which a index
    let out_to_a: Vec<Option<usize>> = out_sub.iter()
        .map(|c| a_sub.iter().position(|ac| ac == c))
        .collect();

    // Find contraction index positions in a (may have multiple for repeated indices)
    let contract_in_a: Vec<(char, usize)> = contraction_indices.iter()
        .flat_map(|&c| {
            a_sub.iter().enumerate()
                .filter(move |(_, ac)| **ac == c)
                .map(move |(pos, _)| (c, pos))
        })
        .collect();

    // Compute dimensions
    let out_dims: Vec<usize> = out_sub.iter().map(|c| dim_map[c]).collect();
    let contract_dims: Vec<usize> = contraction_indices.iter().map(|c| dim_map[c]).collect();
    let out_shape = if out_dims.is_empty() { vec![1] } else { out_dims.clone() };
    let out_strides = compute_strides(&out_shape);
    let contract_size: usize = contract_dims.iter().product();
    let out_size: usize = out_shape.iter().product();

    // Check for repeated indices (diagonal/trace)
    let has_repeated_in_a = {
        let mut seen: HashSet<char> = HashSet::new();
        a_sub.iter().any(|c| !seen.insert(*c))
    };

    let a_strides = a.strides();
    let a_ptr = a.data_ptr();
    let out_ptr = result.data_ptr();
    let a_dtype = a.dtype();

    // Iterate over output positions
    for out_flat in 0..out_size {
        // Decode output position to multi-index
        let mut out_idx = vec![0usize; out_shape.len()];
        let mut rem = out_flat;
        for i in 0..out_shape.len() {
            out_idx[i] = rem / out_strides[i];
            rem %= out_strides[i];
        }

        // Sum over contraction indices
        let mut sum = 0.0f64;

        if contraction_indices.is_empty() {
            // No contraction, just copy/transpose
            // Build mapping from output index position to output value
            let mut out_val_map: HashMap<char, usize> = HashMap::new();
            for (i, &c) in out_sub.iter().enumerate() {
                out_val_map.insert(c, out_idx[i]);
            }

            // Set all positions in a_idx that correspond to output indices
            let mut a_idx = vec![0usize; a_sub.len()];
            for (pos, &c) in a_sub.iter().enumerate() {
                if let Some(&val) = out_val_map.get(&c) {
                    a_idx[pos] = val;
                }
            }

            let a_offset = compute_offset(&a_idx, a_strides);
            let val = unsafe { a_dtype.ops().read_f64(a_ptr, a_offset).unwrap_or(0.0) };
            sum = val;
        } else {
            // Iterate over contraction indices
            for c_flat in 0..contract_size {
                // Decode contraction position
                let mut c_idx = vec![0usize; contract_dims.len()];
                let mut c_rem = c_flat;
                for i in (0..contract_dims.len()).rev() {
                    c_idx[i] = c_rem % contract_dims[i];
                    c_rem /= contract_dims[i];
                }

                // Build full a index
                let mut a_idx = vec![0usize; a_sub.len()];
                for (i, &maybe_a_pos) in out_to_a.iter().enumerate() {
                    if let Some(a_pos) = maybe_a_pos {
                        a_idx[a_pos] = out_idx[i];
                    }
                }
                // Map contraction index to contraction dimension and then to position
                let mut c_idx_map: HashMap<char, usize> = HashMap::new();
                for (ci, &c) in contraction_indices.iter().enumerate() {
                    c_idx_map.insert(c, c_idx[ci]);
                }
                for &(c, a_pos) in &contract_in_a {
                    if let Some(&idx) = c_idx_map.get(&c) {
                        a_idx[a_pos] = idx;
                    }
                }

                // Handle repeated indices
                if has_repeated_in_a {
                    let mut idx_vals: HashMap<char, usize> = HashMap::new();
                    let mut valid = true;
                    for (pos, &c) in a_sub.iter().enumerate() {
                        if let Some(&existing) = idx_vals.get(&c) {
                            if a_idx[pos] != existing {
                                valid = false;
                                break;
                            }
                        } else {
                            idx_vals.insert(c, a_idx[pos]);
                        }
                    }
                    if !valid {
                        continue;
                    }
                }

                let a_offset = compute_offset(&a_idx, a_strides);
                let val = unsafe { a_dtype.ops().read_f64(a_ptr, a_offset).unwrap_or(0.0) };
                sum += val;
            }
        }

        // Write result using dtype-aware method
        let out_dtype = result.dtype();
        unsafe {
            out_dtype.ops().write_f64(out_ptr as *mut u8, out_flat, sum);
        }
    }

    Ok(())
}

fn compute_offset(idx: &[usize], strides: &[isize]) -> isize {
    idx.iter()
        .zip(strides.iter())
        .map(|(&i, &s)| (i as isize) * s)
        .sum()
}

/// Check if binary contraction can use BLAS (no repeated indices within operands).
fn can_use_blas(a_sub: &[char], b_sub: &[char]) -> bool {
    // Check no repeated indices within a
    let mut a_seen: HashSet<char> = HashSet::new();
    for &c in a_sub {
        if !a_seen.insert(c) {
            return false;
        }
    }
    // Check no repeated indices within b
    let mut b_seen: HashSet<char> = HashSet::new();
    for &c in b_sub {
        if !b_seen.insert(c) {
            return false;
        }
    }
    true
}

/// Binary einsum with BLAS optimization.
///
/// For contractions without repeated indices, we can convert to matmul:
/// 1. Permute A so free indices come first, contracted indices last
/// 2. Reshape A to 2D: (product of free, product of contracted)
/// 3. Permute B so contracted indices come first, free indices last
/// 4. Reshape B to 2D: (product of contracted, product of free)
/// 5. Matmul A @ B → (A_free_prod, B_free_prod)
/// 6. Reshape to output shape and permute if needed
fn einsum_binary(
    parsed: &ParsedSubscript,
    a: &RumpyArray,
    b: &RumpyArray,
    result: &RumpyArray,
    dim_map: &HashMap<char, usize>,
    contraction_indices: &[char],
    _output_strides: &[usize],
) -> Result<(), String> {
    let a_sub = &parsed.inputs[0];
    let b_sub = &parsed.inputs[1];
    let out_sub = &parsed.output;

    // Try BLAS path for simple contractions
    if can_use_blas(a_sub, b_sub) && !contraction_indices.is_empty() {
        if let Some(blas_result) = try_blas_contraction(a, b, a_sub, b_sub, out_sub, dim_map, contraction_indices) {
            // Copy result (dtype-aware)
            let out_ptr = result.data_ptr() as *mut u8;
            let blas_ptr = blas_result.data_ptr();
            let itemsize = result.dtype().itemsize();
            let size = result.size() * itemsize;
            unsafe {
                std::ptr::copy_nonoverlapping(blas_ptr, out_ptr, size);
            }
            return Ok(());
        }
    }

    // Fall back to element-wise implementation
    einsum_binary_elementwise(parsed, a, b, result, dim_map, contraction_indices)
}

/// Try to compute binary contraction using BLAS (matmul).
fn try_blas_contraction(
    a: &RumpyArray,
    b: &RumpyArray,
    a_sub: &[char],
    b_sub: &[char],
    out_sub: &[char],
    dim_map: &HashMap<char, usize>,
    contraction_indices: &[char],
) -> Option<RumpyArray> {
    let contract_set: HashSet<char> = contraction_indices.iter().copied().collect();

    // Identify free indices for A (in A but not contracted)
    let a_free: Vec<char> = a_sub.iter().filter(|c| !contract_set.contains(c)).copied().collect();
    let a_contract: Vec<char> = a_sub.iter().filter(|c| contract_set.contains(c)).copied().collect();

    // Identify free indices for B (in B but not contracted)
    let b_free: Vec<char> = b_sub.iter().filter(|c| !contract_set.contains(c)).copied().collect();
    let b_contract: Vec<char> = b_sub.iter().filter(|c| contract_set.contains(c)).copied().collect();

    // Compute dimensions
    let a_free_dims: Vec<usize> = a_free.iter().map(|c| dim_map[c]).collect();
    let a_contract_dims: Vec<usize> = a_contract.iter().map(|c| dim_map[c]).collect();
    let b_free_dims: Vec<usize> = b_free.iter().map(|c| dim_map[c]).collect();
    let b_contract_dims: Vec<usize> = b_contract.iter().map(|c| dim_map[c]).collect();

    let a_free_prod: usize = a_free_dims.iter().product::<usize>().max(1);
    let a_contract_prod: usize = a_contract_dims.iter().product::<usize>().max(1);
    let b_free_prod: usize = b_free_dims.iter().product::<usize>().max(1);
    let b_contract_prod: usize = b_contract_dims.iter().product::<usize>().max(1);

    // Contract dimensions must match
    if a_contract_prod != b_contract_prod {
        return None;
    }

    // Build permutation for A: free indices first, contracted last
    let a_perm: Vec<usize> = a_free.iter()
        .chain(a_contract.iter())
        .map(|c| a_sub.iter().position(|x| x == c).unwrap())
        .collect();

    // Build permutation for B: contracted first, free last
    let b_perm: Vec<usize> = b_contract.iter()
        .chain(b_free.iter())
        .map(|c| b_sub.iter().position(|x| x == c).unwrap())
        .collect();

    // Permute and reshape A to (a_free_prod, a_contract_prod)
    let a_permuted = if a_perm.iter().enumerate().all(|(i, &p)| i == p) {
        a.clone()
    } else {
        a.transpose_axes(&a_perm)
    };
    let a_2d = a_permuted.reshape(vec![a_free_prod, a_contract_prod])?;

    // Permute and reshape B to (b_contract_prod, b_free_prod)
    let b_permuted = if b_perm.iter().enumerate().all(|(i, &p)| i == p) {
        b.clone()
    } else {
        b.transpose_axes(&b_perm)
    };
    let b_2d = b_permuted.reshape(vec![b_contract_prod, b_free_prod])?;

    // Matmul: (a_free_prod, contract_prod) @ (contract_prod, b_free_prod) → (a_free_prod, b_free_prod)
    // matmul uses dtype promotion internally
    let matmul_result = crate::ops::matmul::matmul(&a_2d, &b_2d)?;

    // Build intermediate shape: a_free_dims + b_free_dims
    let mut inter_shape: Vec<usize> = a_free_dims.clone();
    inter_shape.extend(&b_free_dims);

    // Handle scalar output
    if inter_shape.is_empty() {
        return Some(matmul_result);
    }

    let reshaped = matmul_result.reshape(inter_shape.clone())?;

    // Check if output order matches (a_free, b_free)
    let inter_indices: Vec<char> = a_free.iter().chain(b_free.iter()).copied().collect();

    if inter_indices == out_sub {
        // Perfect match, no permutation needed
        Some(reshaped)
    } else {
        // Need to permute to match output order
        let out_perm: Vec<usize> = out_sub.iter()
            .map(|c| inter_indices.iter().position(|x| x == c))
            .collect::<Option<Vec<_>>>()?;
        Some(reshaped.transpose_axes(&out_perm))
    }
}

/// Element-wise binary einsum (fallback for complex cases).
fn einsum_binary_elementwise(
    parsed: &ParsedSubscript,
    a: &RumpyArray,
    b: &RumpyArray,
    result: &RumpyArray,
    dim_map: &HashMap<char, usize>,
    contraction_indices: &[char],
) -> Result<(), String> {
    let a_sub = &parsed.inputs[0];
    let b_sub = &parsed.inputs[1];
    let out_sub = &parsed.output;

    // Compute dimensions
    let out_dims: Vec<usize> = out_sub.iter().map(|c| dim_map[c]).collect();
    let contract_dims: Vec<usize> = contraction_indices.iter().map(|c| dim_map[c]).collect();
    let out_shape = if out_dims.is_empty() { vec![1] } else { out_dims.clone() };
    let out_strides_local = compute_strides(&out_shape);
    let contract_size: usize = contract_dims.iter().product();
    let out_size: usize = out_shape.iter().product();

    // Build index mappings
    let out_to_a: Vec<Option<usize>> = out_sub.iter()
        .map(|c| a_sub.iter().position(|ac| ac == c))
        .collect();
    let out_to_b: Vec<Option<usize>> = out_sub.iter()
        .map(|c| b_sub.iter().position(|bc| bc == c))
        .collect();
    let contract_in_a: Vec<usize> = contraction_indices.iter()
        .filter_map(|c| a_sub.iter().position(|ac| ac == c))
        .collect();
    let contract_in_b: Vec<usize> = contraction_indices.iter()
        .filter_map(|c| b_sub.iter().position(|bc| bc == c))
        .collect();

    let a_strides = a.strides();
    let b_strides = b.strides();
    let a_dtype = a.dtype();
    let b_dtype = b.dtype();

    let a_ptr = a.data_ptr();
    let b_ptr = b.data_ptr();
    let out_ptr = result.data_ptr();

    // Iterate over output positions
    for out_flat in 0..out_size {
        // Decode output position
        let mut out_idx = vec![0usize; out_shape.len()];
        let mut rem = out_flat;
        for i in 0..out_shape.len() {
            out_idx[i] = rem / out_strides_local[i];
            rem %= out_strides_local[i];
        }

        // Sum over contraction indices
        let mut sum = 0.0f64;

        let contract_iters = if contract_size == 0 { 1 } else { contract_size };

        for c_flat in 0..contract_iters {
            // Decode contraction position
            let mut c_idx = vec![0usize; contract_dims.len()];
            if !contract_dims.is_empty() {
                let mut c_rem = c_flat;
                for i in (0..contract_dims.len()).rev() {
                    c_idx[i] = c_rem % contract_dims[i];
                    c_rem /= contract_dims[i];
                }
            }

            // Build a index
            let mut a_idx = vec![0usize; a_sub.len()];
            for (i, &maybe_a_pos) in out_to_a.iter().enumerate() {
                if let Some(a_pos) = maybe_a_pos {
                    a_idx[a_pos] = out_idx[i];
                }
            }
            for (ci, &a_pos) in contract_in_a.iter().enumerate() {
                a_idx[a_pos] = c_idx[ci];
            }

            // Build b index
            let mut b_idx = vec![0usize; b_sub.len()];
            for (i, &maybe_b_pos) in out_to_b.iter().enumerate() {
                if let Some(b_pos) = maybe_b_pos {
                    b_idx[b_pos] = out_idx[i];
                }
            }
            for (ci, &b_pos) in contract_in_b.iter().enumerate() {
                b_idx[b_pos] = c_idx[ci];
            }

            let a_offset = compute_offset(&a_idx, a_strides);
            let b_offset = compute_offset(&b_idx, b_strides);

            let a_val: f64 = unsafe {
                a_dtype.ops().read_f64(a_ptr, a_offset).unwrap_or(0.0)
            };
            let b_val: f64 = unsafe {
                b_dtype.ops().read_f64(b_ptr, b_offset).unwrap_or(0.0)
            };

            sum += a_val * b_val;
        }

        // Write result using dtype-aware method
        let out_dtype = result.dtype();
        unsafe {
            out_dtype.ops().write_f64(out_ptr as *mut u8, out_flat, sum);
        }
    }

    Ok(())
}

/// Multi-operand einsum (3+ operands).
///
/// For now, uses left-to-right pairwise contraction.
/// TODO: Implement optimal path finding.
fn einsum_multi(
    parsed: &ParsedSubscript,
    operands: &[&RumpyArray],
    result: &RumpyArray,
    _dim_map: &HashMap<char, usize>,
    _contraction_indices: &[char],
    _output_strides: &[usize],
) -> Result<(), String> {
    if operands.len() < 2 {
        return Err("Multi-operand einsum requires at least 2 operands".to_string());
    }

    // Simple left-to-right contraction strategy
    // Contract operands[0] with operands[1], then result with operands[2], etc.

    // Build intermediate subscript for first pair
    let sub0 = &parsed.inputs[0];
    let sub1 = &parsed.inputs[1];

    // Compute which indices to keep after first contraction
    // Keep: all indices from sub0 and sub1 that appear in remaining operands or output
    let remaining_indices: HashSet<char> = parsed.inputs[2..]
        .iter()
        .flat_map(|s| s.iter().copied())
        .chain(parsed.output.iter().copied())
        .collect();

    let mut kept_indices: Vec<char> = Vec::new();
    let mut seen: HashSet<char> = HashSet::new();
    for &c in sub0.iter().chain(sub1.iter()) {
        if remaining_indices.contains(&c) && !seen.contains(&c) {
            kept_indices.push(c);
            seen.insert(c);
        }
    }

    // Sort for determinism
    kept_indices.sort();

    // Build intermediate subscript string
    let inter_sub: String = format!(
        "{},{}->{}",
        sub0.iter().collect::<String>(),
        sub1.iter().collect::<String>(),
        kept_indices.iter().collect::<String>()
    );

    // Execute first contraction
    let mut current = einsum(&inter_sub, &[operands[0], operands[1]])?;
    let mut current_sub = kept_indices;

    // Contract with remaining operands
    for (i, op) in operands[2..].iter().enumerate() {
        let op_sub = &parsed.inputs[i + 2];

        // Determine output indices for this step
        let remaining_ops: HashSet<char> = parsed.inputs[i + 3..]
            .iter()
            .flat_map(|s| s.iter().copied())
            .chain(parsed.output.iter().copied())
            .collect();

        let mut next_kept: Vec<char> = Vec::new();
        let mut seen: HashSet<char> = HashSet::new();
        for &c in current_sub.iter().chain(op_sub.iter()) {
            if remaining_ops.contains(&c) && !seen.contains(&c) {
                next_kept.push(c);
                seen.insert(c);
            }
        }

        // If this is the last contraction, use the final output
        let out_indices = if i + 3 >= operands.len() {
            parsed.output.clone()
        } else {
            next_kept.sort();
            next_kept
        };

        let step_sub = format!(
            "{},{}->{}",
            current_sub.iter().collect::<String>(),
            op_sub.iter().collect::<String>(),
            out_indices.iter().collect::<String>()
        );

        current = einsum(&step_sub, &[&current, op])?;
        current_sub = out_indices;
    }

    // Copy result to output array (dtype-aware)
    let out_ptr = result.data_ptr() as *mut u8;
    let cur_ptr = current.data_ptr();
    let itemsize = result.dtype().itemsize();
    let size = result.size() * itemsize;
    unsafe {
        std::ptr::copy_nonoverlapping(cur_ptr, out_ptr, size);
    }

    Ok(())
}

/// Compute optimal contraction path.
///
/// Returns a list of index pairs indicating which operands to contract at each step.
pub fn einsum_path(
    subscript: &str,
    shapes: &[Vec<usize>],
    optimize: &str,
) -> Result<(Vec<(usize, usize)>, String), String> {
    let parsed = parse_subscript(subscript)?;
    let shape_refs: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();
    let dim_map = validate_subscript(&parsed, &shape_refs)?;

    let n = shapes.len();
    if n < 2 {
        return Ok((vec![], "Single operand, no contraction needed".to_string()));
    }

    // Simple left-to-right path for all strategies
    // TODO: Implement actual greedy/optimal algorithms for "greedy"/"optimal" optimize values
    let _ = optimize;
    let path: Vec<(usize, usize)> = (0..n-1)
        .map(|_| (0, 1)) // Always contract first two remaining operands
        .collect();

    let info = format_path_info(&parsed, shapes, &path, &dim_map);
    Ok((path, info))
}

fn format_path_info(
    parsed: &ParsedSubscript,
    shapes: &[Vec<usize>],
    path: &[(usize, usize)],
    dim_map: &HashMap<char, usize>,
) -> String {
    let input_str = parsed.inputs.iter()
        .map(|s| s.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join(",");
    let output_str: String = parsed.output.iter().collect();

    let mut info = format!("  Complete contraction:  {}->{}\n", input_str, output_str);
    info.push_str(&format!("  Number of operands:    {}\n", shapes.len()));
    info.push_str(&format!("  Contraction steps:     {}\n", path.len()));

    // Compute naive FLOP count
    let total_dims: usize = dim_map.values().product();
    info.push_str(&format!("  Naive FLOP count:      ~{:.2e}\n", total_dims as f64 * 2.0));

    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_explicit() {
        let parsed = parse_subscript("ij,jk->ik").unwrap();
        assert_eq!(parsed.inputs.len(), 2);
        assert_eq!(parsed.inputs[0], vec!['i', 'j']);
        assert_eq!(parsed.inputs[1], vec!['j', 'k']);
        assert_eq!(parsed.output, vec!['i', 'k']);
        assert!(parsed.explicit_output);
    }

    #[test]
    fn test_parse_implicit() {
        let parsed = parse_subscript("ij,jk").unwrap();
        assert_eq!(parsed.inputs.len(), 2);
        // Implicit: j appears twice (summed), i and k appear once
        assert_eq!(parsed.output, vec!['i', 'k']);
        assert!(!parsed.explicit_output);
    }

    #[test]
    fn test_parse_trace() {
        let parsed = parse_subscript("ii->").unwrap();
        assert_eq!(parsed.inputs.len(), 1);
        assert_eq!(parsed.inputs[0], vec!['i', 'i']);
        assert!(parsed.output.is_empty());
    }

    #[test]
    fn test_validate_dims() {
        let parsed = parse_subscript("ij,jk->ik").unwrap();
        let shapes = vec![&[2usize, 3][..], &[3, 4][..]];
        let dim_map = validate_subscript(&parsed, &shapes).unwrap();
        assert_eq!(dim_map[&'i'], 2);
        assert_eq!(dim_map[&'j'], 3);
        assert_eq!(dim_map[&'k'], 4);
    }

    #[test]
    fn test_validate_mismatch() {
        let parsed = parse_subscript("ij,jk->ik").unwrap();
        let shapes = vec![&[2usize, 3][..], &[4, 5][..]]; // j mismatch: 3 vs 4
        let result = validate_subscript(&parsed, &shapes);
        assert!(result.is_err());
    }
}
