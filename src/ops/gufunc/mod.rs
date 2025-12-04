//! Generalized universal functions (gufuncs).
//!
//! Gufuncs extend ufuncs to operate on sub-arrays rather than scalars.
//! They support signatures like "(m,n),(n,p)->(m,p)" for matrix multiplication.

pub mod dims;
pub mod signature;

pub use dims::{broadcast_multi_shapes, GufuncDims};
pub use signature::{CoreDims, DimName, GufuncSignature};

use crate::array::{increment_indices, promote_dtype, RumpyArray};

/// Trait for gufunc kernels.
///
/// Implement this trait to define a new gufunc operation.
/// The kernel receives core sub-arrays and writes to output sub-arrays.
pub trait GufuncKernel: Send + Sync {
    /// Returns the signature for this gufunc.
    fn signature(&self) -> &GufuncSignature;

    /// Execute the kernel on core sub-arrays.
    ///
    /// - `inputs`: Read-only core sub-arrays (one per input in signature)
    /// - `outputs`: Mutable core sub-arrays to write results to
    fn call(&self, inputs: &[RumpyArray], outputs: &mut [RumpyArray]);
}

/// Execute a gufunc with the given kernel over input arrays.
///
/// This function:
/// 1. Resolves dimensions from the signature and input shapes
/// 2. Allocates output arrays with the correct shape
/// 3. Iterates over loop dimensions, calling the kernel for each position
/// 4. Returns the output arrays
pub fn gufunc_call<K: GufuncKernel>(
    kernel: &K,
    inputs: &[&RumpyArray],
) -> Option<Vec<RumpyArray>> {
    let sig = kernel.signature();

    // Resolve dimensions
    let dims = GufuncDims::resolve(sig, inputs)?;

    // Determine output dtype (use highest priority input dtype)
    let output_dtype = inputs
        .iter()
        .map(|a| a.dtype())
        .reduce(|a, b| promote_dtype(&a, &b))
        .unwrap_or_else(|| inputs[0].dtype());

    // Allocate outputs
    let outputs: Vec<RumpyArray> = (0..sig.num_outputs())
        .map(|i| {
            let shape = dims.output_shape(i);
            RumpyArray::zeros(shape, output_dtype.clone())
        })
        .collect();

    // Iterate over loop dimensions
    let loop_size = dims.loop_size();
    let mut loop_indices = vec![0usize; dims.loop_shape.len()];

    for _ in 0..loop_size {
        // Extract input core sub-arrays
        let input_cores: Vec<RumpyArray> = inputs
            .iter()
            .enumerate()
            .map(|(i, arr)| arr.gufunc_subarray(&loop_indices, dims.input_loop_ndims[i]))
            .collect();

        // Extract output core sub-arrays
        let output_loop_ndim = dims.loop_shape.len();
        let mut output_cores: Vec<RumpyArray> = outputs
            .iter()
            .map(|arr| arr.gufunc_subarray(&loop_indices, output_loop_ndim))
            .collect();

        // Call kernel
        kernel.call(&input_cores, &mut output_cores);

        // Advance loop indices
        if !dims.loop_shape.is_empty() {
            increment_indices(&mut loop_indices, &dims.loop_shape);
        }
    }

    Some(outputs)
}
