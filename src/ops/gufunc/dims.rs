//! Gufunc dimension resolution.
//!
//! Resolves core dimension sizes from input arrays and computes broadcast loop shape.

use std::collections::HashMap;

use crate::array::{broadcast_shapes, RumpyArray};

use super::signature::{DimName, GufuncSignature};

/// Resolved dimensions for one gufunc call.
#[derive(Debug)]
pub struct GufuncDims {
    /// Broadcast loop shape (shared across all inputs/outputs).
    pub loop_shape: Vec<usize>,

    /// Core dimension sizes, keyed by dimension name.
    pub core_sizes: HashMap<DimName, usize>,

    /// Per-input: number of loop dimensions.
    pub input_loop_ndims: Vec<usize>,

    /// Per-output: resolved core shape.
    pub output_core_shapes: Vec<Vec<usize>>,
}

impl GufuncDims {
    /// Resolve dimensions from signature and actual input arrays.
    ///
    /// Returns None if:
    /// - Input count doesn't match signature
    /// - Arrays have fewer dimensions than required core dims
    /// - Core dimensions with same name have different sizes
    /// - Loop dimensions don't broadcast
    pub fn resolve(sig: &GufuncSignature, inputs: &[&RumpyArray]) -> Option<Self> {
        if inputs.len() != sig.num_inputs() {
            return None;
        }

        let mut loop_shapes: Vec<Vec<usize>> = vec![];
        let mut core_sizes: HashMap<DimName, usize> = HashMap::new();
        let mut input_loop_ndims: Vec<usize> = vec![];

        // Process each input
        for (i, arr) in inputs.iter().enumerate() {
            let core_ndim = sig.input_core_ndim(i);
            let total_ndim = arr.ndim();

            if total_ndim < core_ndim {
                return None; // Not enough dimensions
            }

            let loop_ndim = total_ndim - core_ndim;
            let loop_shape = &arr.shape()[..loop_ndim];
            let core_shape = &arr.shape()[loop_ndim..];

            loop_shapes.push(loop_shape.to_vec());
            input_loop_ndims.push(loop_ndim);

            // Check and record core dimension sizes
            for (j, dim_name) in sig.inputs[i].dims.iter().enumerate() {
                let size = core_shape[j];
                if let Some(&existing) = core_sizes.get(dim_name) {
                    if existing != size {
                        return None; // Dimension mismatch
                    }
                } else {
                    core_sizes.insert(dim_name.clone(), size);
                }
            }
        }

        // Broadcast loop shapes together
        let loop_shape = broadcast_multi_shapes(&loop_shapes)?;

        // Compute output core shapes from signature
        let output_core_shapes: Vec<Vec<usize>> = sig
            .outputs
            .iter()
            .map(|out_dims| {
                out_dims
                    .dims
                    .iter()
                    .map(|name| {
                        *core_sizes.get(name).unwrap_or(&1) // Default to 1 for unbound dims
                    })
                    .collect()
            })
            .collect();

        Some(GufuncDims {
            loop_shape,
            core_sizes,
            input_loop_ndims,
            output_core_shapes,
        })
    }

    /// Total number of kernel invocations (product of loop dimensions).
    pub fn loop_size(&self) -> usize {
        if self.loop_shape.is_empty() {
            1
        } else {
            self.loop_shape.iter().product()
        }
    }

    /// Full output shape for output i (loop_shape + core_shape).
    pub fn output_shape(&self, i: usize) -> Vec<usize> {
        let mut shape = self.loop_shape.clone();
        shape.extend(&self.output_core_shapes[i]);
        shape
    }
}

/// Broadcast multiple shapes together.
pub fn broadcast_multi_shapes(shapes: &[Vec<usize>]) -> Option<Vec<usize>> {
    if shapes.is_empty() {
        return Some(vec![]);
    }
    let mut result = shapes[0].clone();
    for shape in &shapes[1..] {
        result = broadcast_shapes(&result, shape)?;
    }
    Some(result)
}
