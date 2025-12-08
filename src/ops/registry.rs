//! UFunc registry for type-specific inner loops.
//!
//! Most operations are now handled by the kernel/dispatch system.
//! This registry is kept for:
//! - Reduce loops for Bool (niche type)

use crate::array::dtype::{DTypeKind, ReduceOp};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Type signature for a ufunc loop.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeSignature {
    pub inputs: Vec<DTypeKind>,
    pub output: DTypeKind,
}

impl TypeSignature {
    pub fn reduce(input: DTypeKind, out: DTypeKind) -> Self {
        Self {
            inputs: vec![input],
            output: out,
        }
    }
}

/// Initialize accumulator for reduction.
pub type ReduceInitFn = unsafe fn(out_ptr: *mut u8, idx: usize);

/// Accumulate one element into reduction (legacy, for full-array reductions).
pub type ReduceAccFn = unsafe fn(
    acc_ptr: *mut u8,
    idx: usize,
    val_ptr: *const u8,
    byte_offset: isize,
);

/// Strided reduction loop - processes N elements into an accumulator.
///
/// This is the fast path for axis reductions. Handles both contiguous
/// (stride == itemsize) and strided cases with SIMD-friendly inner loops.
///
/// # Safety
/// Caller must ensure:
/// - `acc_ptr` points to valid, initialized accumulator
/// - `src_ptr` is valid for `n` elements at `stride` intervals
pub type ReduceLoopFn = unsafe fn(
    acc_ptr: *mut u8,      // Single accumulator element
    src_ptr: *const u8,    // Source data start
    n: usize,              // Number of elements to reduce
    stride: isize,         // Byte stride between elements
);

/// Registry for ufunc inner loops.
///
/// Note: Most operations are now handled by the kernel/dispatch system
/// in `dispatch.rs`. This registry is kept only for:
/// - Reduce loops for Bool (niche type)
pub struct UFuncRegistry {
    reduce_loops: HashMap<(ReduceOp, TypeSignature), (ReduceInitFn, ReduceAccFn)>,
    /// Strided reduction loops for axis reductions (fast path)
    reduce_strided_loops: HashMap<(ReduceOp, TypeSignature), (ReduceInitFn, ReduceLoopFn)>,
}

impl UFuncRegistry {
    pub fn new() -> Self {
        Self {
            reduce_loops: HashMap::new(),
            reduce_strided_loops: HashMap::new(),
        }
    }

    /// Register a reduce loop for a specific operation and type signature.
    pub fn register_reduce(
        &mut self,
        op: ReduceOp,
        sig: TypeSignature,
        init: ReduceInitFn,
        acc: ReduceAccFn,
    ) {
        self.reduce_loops.insert((op, sig), (init, acc));
    }

    /// Look up a reduce loop. Returns None if no registered loop exists.
    pub fn lookup_reduce(
        &self,
        op: ReduceOp,
        input: DTypeKind,
    ) -> Option<(ReduceInitFn, ReduceAccFn, DTypeKind)> {
        let sig = TypeSignature::reduce(input.clone(), input.clone());
        self.reduce_loops.get(&(op, sig)).map(|&(init, acc)| (init, acc, input))
    }

    /// Register a strided reduce loop for axis reductions.
    pub fn register_reduce_strided(
        &mut self,
        op: ReduceOp,
        sig: TypeSignature,
        init: ReduceInitFn,
        loop_fn: ReduceLoopFn,
    ) {
        self.reduce_strided_loops.insert((op, sig), (init, loop_fn));
    }

    /// Look up a strided reduce loop. Returns None if no registered loop exists.
    pub fn lookup_reduce_strided(
        &self,
        op: ReduceOp,
        input: DTypeKind,
    ) -> Option<(ReduceInitFn, ReduceLoopFn, DTypeKind)> {
        let sig = TypeSignature::reduce(input.clone(), input.clone());
        self.reduce_strided_loops.get(&(op, sig)).map(|&(init, loop_fn)| (init, loop_fn, input))
    }
}

impl Default for UFuncRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Global registry
static REGISTRY: OnceLock<RwLock<UFuncRegistry>> = OnceLock::new();

/// Get reference to the global registry.
pub fn registry() -> &'static RwLock<UFuncRegistry> {
    REGISTRY.get_or_init(|| RwLock::new(init_default_loops()))
}

/// Initialize default loops for Bool.
///
/// All other types are handled by the kernel/dispatch system.
fn init_default_loops() -> UFuncRegistry {
    let mut reg = UFuncRegistry::new();

    // Bool reduce loops (Sum=any, Prod=all)
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Bool, DTypeKind::Bool),
        |out_ptr, idx| unsafe { *out_ptr.add(idx) = 0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = acc_ptr.add(idx);
            let v = *val_ptr.offset(byte_offset);
            if v != 0 { *acc = 1; }
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Bool, DTypeKind::Bool),
        |out_ptr, idx| unsafe { *out_ptr.add(idx) = 1; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = acc_ptr.add(idx);
            let v = *val_ptr.offset(byte_offset);
            if v == 0 { *acc = 0; }
        },
    );

    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_signature_equality() {
        let sig1 = TypeSignature::reduce(DTypeKind::Float64, DTypeKind::Float64);
        let sig2 = TypeSignature::reduce(DTypeKind::Float64, DTypeKind::Float64);
        let sig3 = TypeSignature::reduce(DTypeKind::Float32, DTypeKind::Float32);

        assert_eq!(sig1, sig2);
        assert_ne!(sig1, sig3);
    }

    #[test]
    fn test_bool_sum_loop() {
        let reg = init_default_loops();
        let (init_fn, acc_fn, _) = reg
            .lookup_reduce(ReduceOp::Sum, DTypeKind::Bool)
            .unwrap();

        let values: [u8; 4] = [0, 1, 0, 1];
        let mut acc: u8 = 0;

        unsafe {
            init_fn(&mut acc as *mut u8, 0);
            for v in &values {
                acc_fn(
                    &mut acc as *mut u8,
                    0,
                    v as *const u8,
                    0,
                );
            }
        }

        assert_eq!(acc, 1); // any=true
    }
}
