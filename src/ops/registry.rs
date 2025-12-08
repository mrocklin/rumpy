//! UFunc registry for type-specific inner loops.
//!
//! Provides NumPy-style dispatch: registered loops checked first,
//! then fallback to DTypeOps trait methods.

use crate::array::dtype::{BinaryOp, BitwiseOp, DTypeKind, ReduceOp, UnaryOp};
use half::f16;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Type signature for a ufunc loop.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeSignature {
    pub inputs: Vec<DTypeKind>,
    pub output: DTypeKind,
}

impl TypeSignature {
    pub fn binary(a: DTypeKind, b: DTypeKind, out: DTypeKind) -> Self {
        Self {
            inputs: vec![a, b],
            output: out,
        }
    }

    pub fn unary(input: DTypeKind, out: DTypeKind) -> Self {
        Self {
            inputs: vec![input],
            output: out,
        }
    }

    pub fn reduce(input: DTypeKind, out: DTypeKind) -> Self {
        Self {
            inputs: vec![input],
            output: out,
        }
    }
}

/// Inner loop function for binary operations.
///
/// Processes `n` elements, advancing pointers by strides.
/// The loop handles both contiguous (stride=itemsize) and strided cases.
///
/// # Safety
/// Caller must ensure pointers are valid for `n` elements with given strides.
pub type BinaryLoopFn = unsafe fn(
    a_ptr: *const u8,
    b_ptr: *const u8,
    out_ptr: *mut u8,
    n: usize,
    strides: (isize, isize, isize),  // (a_stride, b_stride, out_stride)
);

/// Inner loop function for unary operations.
///
/// Processes `n` elements, advancing pointers by strides.
///
/// # Safety
/// Caller must ensure pointers are valid for `n` elements with given strides.
pub type UnaryLoopFn = unsafe fn(
    src_ptr: *const u8,
    out_ptr: *mut u8,
    n: usize,
    strides: (isize, isize),  // (src_stride, out_stride)
);

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
pub struct UFuncRegistry {
    binary_loops: HashMap<(BinaryOp, TypeSignature), BinaryLoopFn>,
    unary_loops: HashMap<(UnaryOp, TypeSignature), UnaryLoopFn>,
    reduce_loops: HashMap<(ReduceOp, TypeSignature), (ReduceInitFn, ReduceAccFn)>,
    /// Strided reduction loops for axis reductions (fast path)
    reduce_strided_loops: HashMap<(ReduceOp, TypeSignature), (ReduceInitFn, ReduceLoopFn)>,
    /// Bitwise binary loops (integer/bool types)
    bitwise_binary_loops: HashMap<(BitwiseOp, DTypeKind), BinaryLoopFn>,
    /// Bitwise NOT loops (integer/bool types)
    bitwise_not_loops: HashMap<DTypeKind, UnaryLoopFn>,
}

impl UFuncRegistry {
    pub fn new() -> Self {
        Self {
            binary_loops: HashMap::new(),
            unary_loops: HashMap::new(),
            reduce_loops: HashMap::new(),
            reduce_strided_loops: HashMap::new(),
            bitwise_binary_loops: HashMap::new(),
            bitwise_not_loops: HashMap::new(),
        }
    }

    /// Register a binary loop for a specific operation and type signature.
    pub fn register_binary(&mut self, op: BinaryOp, sig: TypeSignature, f: BinaryLoopFn) {
        self.binary_loops.insert((op, sig), f);
    }

    /// Register a unary loop for a specific operation and type signature.
    pub fn register_unary(&mut self, op: UnaryOp, sig: TypeSignature, f: UnaryLoopFn) {
        self.unary_loops.insert((op, sig), f);
    }

    /// Look up a binary loop. Returns None if no registered loop exists.
    pub fn lookup_binary(
        &self,
        op: BinaryOp,
        a: DTypeKind,
        b: DTypeKind,
    ) -> Option<(BinaryLoopFn, DTypeKind)> {
        // For now, only check exact same-type matches (a == b == out)
        let sig = TypeSignature::binary(a.clone(), b.clone(), a.clone());
        if a == b {
            self.binary_loops.get(&(op, sig)).map(|&f| (f, a))
        } else {
            None
        }
    }

    /// Look up a unary loop. Returns None if no registered loop exists.
    pub fn lookup_unary(
        &self,
        op: UnaryOp,
        input: DTypeKind,
    ) -> Option<(UnaryLoopFn, DTypeKind)> {
        let sig = TypeSignature::unary(input.clone(), input.clone());
        self.unary_loops.get(&(op, sig)).map(|&f| (f, input))
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

    /// Register a bitwise binary loop for a specific operation and dtype.
    pub fn register_bitwise_binary(&mut self, op: BitwiseOp, dtype: DTypeKind, f: BinaryLoopFn) {
        self.bitwise_binary_loops.insert((op, dtype), f);
    }

    /// Look up a bitwise binary loop.
    pub fn lookup_bitwise_binary(&self, op: BitwiseOp, dtype: DTypeKind) -> Option<BinaryLoopFn> {
        self.bitwise_binary_loops.get(&(op, dtype)).copied()
    }

    /// Register a bitwise NOT loop for a specific dtype.
    pub fn register_bitwise_not(&mut self, dtype: DTypeKind, f: UnaryLoopFn) {
        self.bitwise_not_loops.insert(dtype, f);
    }

    /// Look up a bitwise NOT loop.
    pub fn lookup_bitwise_not(&self, dtype: DTypeKind) -> Option<UnaryLoopFn> {
        self.bitwise_not_loops.get(&dtype).copied()
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

/// Initialize default loops for common types.
fn init_default_loops() -> UFuncRegistry {
    let mut reg = UFuncRegistry::new();

    // Macro for strided binary loops with contiguous fast path
    macro_rules! register_strided_binary {
        ($reg:expr, $op:expr, $kind:expr, $T:ty, $f:expr) => {
            $reg.register_binary(
                $op,
                TypeSignature::binary($kind.clone(), $kind.clone(), $kind.clone()),
                |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    let (sa, sb, so) = strides;
                    if sa == itemsize && sb == itemsize && so == itemsize {
                        // Contiguous fast path - LLVM can auto-vectorize
                        let a = std::slice::from_raw_parts(a_ptr as *const $T, n);
                        let b = std::slice::from_raw_parts(b_ptr as *const $T, n);
                        let out = std::slice::from_raw_parts_mut(out_ptr as *mut $T, n);
                        for i in 0..n {
                            out[i] = $f(a[i], b[i]);
                        }
                    } else {
                        // Strided path
                        let mut ap = a_ptr;
                        let mut bp = b_ptr;
                        let mut op = out_ptr;
                        for _ in 0..n {
                            let a = *(ap as *const $T);
                            let b = *(bp as *const $T);
                            *(op as *mut $T) = $f(a, b);
                            ap = ap.offset(sa);
                            bp = bp.offset(sb);
                            op = op.offset(so);
                        }
                    }
                },
            );
        };
    }

    // Register all arithmetic ops for a numeric type
    macro_rules! register_arithmetic {
        ($reg:expr, $kind:expr, $T:ty) => {
            register_strided_binary!($reg, BinaryOp::Add, $kind, $T, |a, b| a + b);
            register_strided_binary!($reg, BinaryOp::Sub, $kind, $T, |a, b| a - b);
            register_strided_binary!($reg, BinaryOp::Mul, $kind, $T, |a, b| a * b);
            register_strided_binary!($reg, BinaryOp::Div, $kind, $T, |a: $T, b: $T| a / b);
            register_strided_binary!($reg, BinaryOp::Mod, $kind, $T, |a: $T, b: $T| a % b);
        };
    }

    // Float-specific ops (pow, floordiv)
    macro_rules! register_float_binary {
        ($reg:expr, $kind:expr, $T:ty) => {
            register_strided_binary!($reg, BinaryOp::Pow, $kind, $T, |a: $T, b: $T| a.powf(b));
            register_strided_binary!($reg, BinaryOp::FloorDiv, $kind, $T, |a: $T, b: $T| (a / b).floor());
        };
    }

    // Stream 2: Binary math ops (float-only)
    macro_rules! register_stream2_binary {
        ($reg:expr, $kind:expr, $T:ty) => {
            register_strided_binary!($reg, BinaryOp::Arctan2, $kind, $T, |a: $T, b: $T| a.atan2(b));
            register_strided_binary!($reg, BinaryOp::Hypot, $kind, $T, |a: $T, b: $T| a.hypot(b));
            register_strided_binary!($reg, BinaryOp::Copysign, $kind, $T, |a: $T, b: $T| a.copysign(b));
            register_strided_binary!($reg, BinaryOp::FMax, $kind, $T, |a: $T, b: $T| if b.is_nan() { a } else if a.is_nan() { b } else { a.max(b) });
            register_strided_binary!($reg, BinaryOp::FMin, $kind, $T, |a: $T, b: $T| if b.is_nan() { a } else if a.is_nan() { b } else { a.min(b) });
            register_strided_binary!($reg, BinaryOp::Logaddexp, $kind, $T, |a: $T, b: $T| {
                let m = a.max(b);
                if m.is_infinite() { m } else { m + (1.0 as $T + (-(a - b).abs()).exp()).ln() }
            });
            register_strided_binary!($reg, BinaryOp::Logaddexp2, $kind, $T, |a: $T, b: $T| {
                let m = a.max(b);
                let ln2 = std::f64::consts::LN_2 as $T;
                if m.is_infinite() { m } else { m + ((1.0 as $T + (-(a - b).abs() * ln2).exp()).ln() / ln2) }
            });
        };
    }

    // Binary loops for all numeric types
    register_arithmetic!(reg, DTypeKind::Float64, f64);
    register_arithmetic!(reg, DTypeKind::Float32, f32);
    register_arithmetic!(reg, DTypeKind::Int64, i64);
    register_arithmetic!(reg, DTypeKind::Int32, i32);
    register_arithmetic!(reg, DTypeKind::Int16, i16);
    register_arithmetic!(reg, DTypeKind::Uint64, u64);
    register_arithmetic!(reg, DTypeKind::Uint32, u32);
    register_arithmetic!(reg, DTypeKind::Uint16, u16);
    register_arithmetic!(reg, DTypeKind::Uint8, u8);

    // Float-specific binary ops (pow, floordiv)
    register_float_binary!(reg, DTypeKind::Float64, f64);
    register_float_binary!(reg, DTypeKind::Float32, f32);

    // Stream 2: Binary math ops (arctan2, hypot, fmax, fmin, copysign, logaddexp, logaddexp2)
    register_stream2_binary!(reg, DTypeKind::Float64, f64);
    register_stream2_binary!(reg, DTypeKind::Float32, f32);

    // Float16 binary loops (convert to f32 for ops)
    macro_rules! register_f16_binary {
        ($reg:expr, $op:expr, $f:expr) => {
            $reg.register_binary(
                $op,
                TypeSignature::binary(DTypeKind::Float16, DTypeKind::Float16, DTypeKind::Float16),
                |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
                    let (sa, sb, so) = strides;
                    let mut ap = a_ptr;
                    let mut bp = b_ptr;
                    let mut op = out_ptr;
                    for _ in 0..n {
                        let a = (*(ap as *const f16)).to_f32();
                        let b = (*(bp as *const f16)).to_f32();
                        *(op as *mut f16) = f16::from_f32($f(a, b));
                        ap = ap.offset(sa);
                        bp = bp.offset(sb);
                        op = op.offset(so);
                    }
                },
            );
        };
    }
    register_f16_binary!(reg, BinaryOp::Add, |a: f32, b: f32| a + b);
    register_f16_binary!(reg, BinaryOp::Sub, |a: f32, b: f32| a - b);
    register_f16_binary!(reg, BinaryOp::Mul, |a: f32, b: f32| a * b);
    register_f16_binary!(reg, BinaryOp::Div, |a: f32, b: f32| a / b);
    register_f16_binary!(reg, BinaryOp::Mod, |a: f32, b: f32| a % b);
    register_f16_binary!(reg, BinaryOp::Pow, |a: f32, b: f32| a.powf(b));
    register_f16_binary!(reg, BinaryOp::FloorDiv, |a: f32, b: f32| (a / b).floor());
    // Stream 2: Float16 binary math ops
    register_f16_binary!(reg, BinaryOp::Arctan2, |a: f32, b: f32| a.atan2(b));
    register_f16_binary!(reg, BinaryOp::Hypot, |a: f32, b: f32| a.hypot(b));
    register_f16_binary!(reg, BinaryOp::Copysign, |a: f32, b: f32| a.copysign(b));
    register_f16_binary!(reg, BinaryOp::FMax, |a: f32, b: f32| if b.is_nan() { a } else if a.is_nan() { b } else { a.max(b) });
    register_f16_binary!(reg, BinaryOp::FMin, |a: f32, b: f32| if b.is_nan() { a } else if a.is_nan() { b } else { a.min(b) });
    register_f16_binary!(reg, BinaryOp::Logaddexp, |a: f32, b: f32| {
        let m = a.max(b);
        if m.is_infinite() { m } else { m + (1.0_f32 + (-(a - b).abs()).exp()).ln() }
    });
    register_f16_binary!(reg, BinaryOp::Logaddexp2, |a: f32, b: f32| {
        let m = a.max(b);
        let ln2 = std::f32::consts::LN_2;
        if m.is_infinite() { m } else { m + ((1.0_f32 + (-(a - b).abs() * ln2).exp()).ln() / ln2) }
    });
    register_f16_binary!(reg, BinaryOp::Nextafter, |a: f32, b: f32| {
        if a.is_nan() || b.is_nan() { f32::NAN }
        else if a == b { b }
        else if a < b { f32::from_bits(if a >= 0.0 { a.to_bits() + 1 } else { a.to_bits() - 1 }) }
        else { f32::from_bits(if a > 0.0 { a.to_bits() - 1 } else { a.to_bits() + 1 }) }
    });

    // Bool binary loops (Add=or, Mul=and)
    reg.register_binary(
        BinaryOp::Add,
        TypeSignature::binary(DTypeKind::Bool, DTypeKind::Bool, DTypeKind::Bool),
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            let mut ap = a_ptr;
            let mut bp = b_ptr;
            let mut op = out_ptr;
            for _ in 0..n {
                let a = *ap != 0;
                let b = *bp != 0;
                *op = (a || b) as u8;
                ap = ap.offset(sa);
                bp = bp.offset(sb);
                op = op.offset(so);
            }
        },
    );
    reg.register_binary(
        BinaryOp::Mul,
        TypeSignature::binary(DTypeKind::Bool, DTypeKind::Bool, DTypeKind::Bool),
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            let mut ap = a_ptr;
            let mut bp = b_ptr;
            let mut op = out_ptr;
            for _ in 0..n {
                let a = *ap != 0;
                let b = *bp != 0;
                *op = (a && b) as u8;
                ap = ap.offset(sa);
                bp = bp.offset(sb);
                op = op.offset(so);
            }
        },
    );

    // Complex binary/unary/reduce loops - macro to reduce duplication
    macro_rules! register_complex_loops {
        ($reg:expr, $kind:expr, $T:ty) => {
            // Binary ops
            $reg.register_binary(
                BinaryOp::Add,
                TypeSignature::binary($kind.clone(), $kind.clone(), $kind.clone()),
                |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
                    let (sa, sb, so) = strides;
                    let (mut ap, mut bp, mut op) = (a_ptr, b_ptr, out_ptr);
                    for _ in 0..n {
                        let (a, b, out) = (ap as *const $T, bp as *const $T, op as *mut $T);
                        *out = *a + *b;
                        *out.add(1) = *a.add(1) + *b.add(1);
                        ap = ap.offset(sa); bp = bp.offset(sb); op = op.offset(so);
                    }
                },
            );
            $reg.register_binary(
                BinaryOp::Sub,
                TypeSignature::binary($kind.clone(), $kind.clone(), $kind.clone()),
                |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
                    let (sa, sb, so) = strides;
                    let (mut ap, mut bp, mut op) = (a_ptr, b_ptr, out_ptr);
                    for _ in 0..n {
                        let (a, b, out) = (ap as *const $T, bp as *const $T, op as *mut $T);
                        *out = *a - *b;
                        *out.add(1) = *a.add(1) - *b.add(1);
                        ap = ap.offset(sa); bp = bp.offset(sb); op = op.offset(so);
                    }
                },
            );
            $reg.register_binary(
                BinaryOp::Mul,
                TypeSignature::binary($kind.clone(), $kind.clone(), $kind.clone()),
                |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
                    let (sa, sb, so) = strides;
                    let (mut ap, mut bp, mut op) = (a_ptr, b_ptr, out_ptr);
                    for _ in 0..n {
                        let (a, b) = (ap as *const $T, bp as *const $T);
                        let (ar, ai, br, bi) = (*a, *a.add(1), *b, *b.add(1));
                        let out = op as *mut $T;
                        *out = ar * br - ai * bi;
                        *out.add(1) = ar * bi + ai * br;
                        ap = ap.offset(sa); bp = bp.offset(sb); op = op.offset(so);
                    }
                },
            );
            $reg.register_binary(
                BinaryOp::Div,
                TypeSignature::binary($kind.clone(), $kind.clone(), $kind.clone()),
                |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
                    let (sa, sb, so) = strides;
                    let (mut ap, mut bp, mut op) = (a_ptr, b_ptr, out_ptr);
                    for _ in 0..n {
                        let (a, b) = (ap as *const $T, bp as *const $T);
                        let (ar, ai, br, bi) = (*a, *a.add(1), *b, *b.add(1));
                        let denom = br * br + bi * bi;
                        let out = op as *mut $T;
                        if denom != 0.0 {
                            *out = (ar * br + ai * bi) / denom;
                            *out.add(1) = (ai * br - ar * bi) / denom;
                        } else {
                            *out = <$T>::NAN;
                            *out.add(1) = <$T>::NAN;
                        }
                        ap = ap.offset(sa); bp = bp.offset(sb); op = op.offset(so);
                    }
                },
            );
            // Unary neg
            $reg.register_unary(
                UnaryOp::Neg,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let (src, out) = (sp as *const $T, op as *mut $T);
                        *out = -(*src);
                        *out.add(1) = -(*src.add(1));
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Unary abs - complex abs returns |z| = sqrt(r² + i²)
            $reg.register_unary(
                UnaryOp::Abs,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let src = sp as *const $T;
                        let (r, i) = (*src, *src.add(1));
                        let mag = (r * r + i * i).sqrt();
                        let out = op as *mut $T;
                        *out = mag;
                        *out.add(1) = 0.0;  // Result is real
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Unary sqrt - complex sqrt
            $reg.register_unary(
                UnaryOp::Sqrt,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let src = sp as *const $T;
                        let (r, i) = (*src as f64, *src.add(1) as f64);
                        let mag = (r * r + i * i).sqrt();
                        let out = op as *mut $T;
                        if mag == 0.0 {
                            *out = 0.0;
                            *out.add(1) = 0.0;
                        } else {
                            // sqrt(z) = sqrt((|z| + Re(z))/2) + i*sign(Im(z))*sqrt((|z| - Re(z))/2)
                            let sqrt_mag = mag.sqrt();
                            let half_theta = i.atan2(r) / 2.0;
                            *out = (sqrt_mag * half_theta.cos()) as $T;
                            *out.add(1) = (sqrt_mag * half_theta.sin()) as $T;
                        }
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Unary exp - complex exp: e^(a+bi) = e^a * (cos(b) + i*sin(b))
            $reg.register_unary(
                UnaryOp::Exp,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let src = sp as *const $T;
                        let (r, i) = (*src as f64, *src.add(1) as f64);
                        let exp_r = r.exp();
                        let out = op as *mut $T;
                        *out = (exp_r * i.cos()) as $T;
                        *out.add(1) = (exp_r * i.sin()) as $T;
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Unary log - complex log: ln(z) = ln|z| + i*arg(z)
            $reg.register_unary(
                UnaryOp::Log,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let src = sp as *const $T;
                        let (r, i) = (*src as f64, *src.add(1) as f64);
                        let mag = (r * r + i * i).sqrt();
                        let out = op as *mut $T;
                        *out = mag.ln() as $T;
                        *out.add(1) = i.atan2(r) as $T;
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Unary sin - complex sin: sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
            $reg.register_unary(
                UnaryOp::Sin,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let src = sp as *const $T;
                        let (r, i) = (*src as f64, *src.add(1) as f64);
                        let out = op as *mut $T;
                        *out = (r.sin() * i.cosh()) as $T;
                        *out.add(1) = (r.cos() * i.sinh()) as $T;
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Unary cos - complex cos: cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
            $reg.register_unary(
                UnaryOp::Cos,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let src = sp as *const $T;
                        let (r, i) = (*src as f64, *src.add(1) as f64);
                        let out = op as *mut $T;
                        *out = (r.cos() * i.cosh()) as $T;
                        *out.add(1) = (-(r.sin() * i.sinh())) as $T;
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Unary tan - complex tan: tan(z) = sin(z)/cos(z)
            $reg.register_unary(
                UnaryOp::Tan,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let src = sp as *const $T;
                        let (r, i) = (*src as f64, *src.add(1) as f64);
                        // tan(z) = sin(z)/cos(z) = (sin(r)cosh(i) + i*cos(r)sinh(i)) / (cos(r)cosh(i) - i*sin(r)sinh(i))
                        let sin_r = r.sin();
                        let cos_r = r.cos();
                        let sinh_i = i.sinh();
                        let cosh_i = i.cosh();
                        let (nr, ni) = (sin_r * cosh_i, cos_r * sinh_i);
                        let (dr, di) = (cos_r * cosh_i, -sin_r * sinh_i);
                        let denom = dr * dr + di * di;
                        let out = op as *mut $T;
                        if denom != 0.0 {
                            *out = ((nr * dr + ni * di) / denom) as $T;
                            *out.add(1) = ((ni * dr - nr * di) / denom) as $T;
                        } else {
                            *out = <$T>::NAN;
                            *out.add(1) = <$T>::NAN;
                        }
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Unary sinh - complex sinh: sinh(a+bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
            $reg.register_unary(
                UnaryOp::Sinh,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let src = sp as *const $T;
                        let (r, i) = (*src as f64, *src.add(1) as f64);
                        let out = op as *mut $T;
                        *out = (r.sinh() * i.cos()) as $T;
                        *out.add(1) = (r.cosh() * i.sin()) as $T;
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Unary cosh - complex cosh: cosh(a+bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
            $reg.register_unary(
                UnaryOp::Cosh,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let src = sp as *const $T;
                        let (r, i) = (*src as f64, *src.add(1) as f64);
                        let out = op as *mut $T;
                        *out = (r.cosh() * i.cos()) as $T;
                        *out.add(1) = (r.sinh() * i.sin()) as $T;
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Unary tanh - complex tanh
            $reg.register_unary(
                UnaryOp::Tanh,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let (mut sp, mut op) = (src_ptr, out_ptr);
                    for _ in 0..n {
                        let src = sp as *const $T;
                        let (r, i) = (*src as f64, *src.add(1) as f64);
                        // tanh(z) = sinh(z)/cosh(z)
                        let (sinh_r, cosh_r) = (r.sinh(), r.cosh());
                        let (sin_i, cos_i) = (i.sin(), i.cos());
                        let (nr, ni) = (sinh_r * cos_i, cosh_r * sin_i);
                        let (dr, di) = (cosh_r * cos_i, sinh_r * sin_i);
                        let denom = dr * dr + di * di;
                        let out = op as *mut $T;
                        if denom != 0.0 {
                            *out = ((nr * dr + ni * di) / denom) as $T;
                            *out.add(1) = ((ni * dr - nr * di) / denom) as $T;
                        } else {
                            *out = <$T>::NAN;
                            *out.add(1) = <$T>::NAN;
                        }
                        sp = sp.offset(ss); op = op.offset(so);
                    }
                },
            );
            // Reduce ops
            $reg.register_reduce(
                ReduceOp::Sum,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, idx| unsafe {
                    let out = (out_ptr as *mut $T).add(idx * 2);
                    *out = 0.0; *out.add(1) = 0.0;
                },
                |acc_ptr, idx, val_ptr, byte_offset| unsafe {
                    let acc = (acc_ptr as *mut $T).add(idx * 2);
                    let v = val_ptr.offset(byte_offset) as *const $T;
                    *acc += *v; *acc.add(1) += *v.add(1);
                },
            );
            $reg.register_reduce(
                ReduceOp::Prod,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, idx| unsafe {
                    let out = (out_ptr as *mut $T).add(idx * 2);
                    *out = 1.0; *out.add(1) = 0.0;
                },
                |acc_ptr, idx, val_ptr, byte_offset| unsafe {
                    let acc = (acc_ptr as *mut $T).add(idx * 2);
                    let v = val_ptr.offset(byte_offset) as *const $T;
                    let (ar, ai, vr, vi) = (*acc, *acc.add(1), *v, *v.add(1));
                    *acc = ar * vr - ai * vi;
                    *acc.add(1) = ar * vi + ai * vr;
                },
            );
            // Strided reduce for Sum (used in axis reductions)
            $reg.register_reduce_strided(
                ReduceOp::Sum,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, _idx| unsafe {
                    let out = out_ptr as *mut $T;
                    *out = 0.0; *out.add(1) = 0.0;
                },
                |acc_ptr, src_ptr, n, stride| unsafe {
                    let itemsize = std::mem::size_of::<$T>() as isize * 2;  // Complex has 2 components
                    let acc = acc_ptr as *mut $T;
                    if stride == itemsize {
                        // Contiguous: use slice for vectorization
                        let src = std::slice::from_raw_parts(src_ptr as *const $T, n * 2);
                        let (mut sum_r, mut sum_i) = (*acc, *acc.add(1));
                        for i in (0..n*2).step_by(2) {
                            sum_r += src[i];
                            sum_i += src[i + 1];
                        }
                        *acc = sum_r;
                        *acc.add(1) = sum_i;
                    } else {
                        // Strided
                        let mut p = src_ptr;
                        for _ in 0..n {
                            let v = p as *const $T;
                            *acc += *v;
                            *acc.add(1) += *v.add(1);
                            p = p.offset(stride);
                        }
                    }
                },
            );
            // Strided reduce for Prod
            $reg.register_reduce_strided(
                ReduceOp::Prod,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, _idx| unsafe {
                    let out = out_ptr as *mut $T;
                    *out = 1.0; *out.add(1) = 0.0;
                },
                |acc_ptr, src_ptr, n, stride| unsafe {
                    let itemsize = std::mem::size_of::<$T>() as isize * 2;
                    let acc = acc_ptr as *mut $T;
                    if stride == itemsize {
                        // Contiguous
                        let src = std::slice::from_raw_parts(src_ptr as *const $T, n * 2);
                        let (mut pr, mut pi) = (*acc, *acc.add(1));
                        for i in (0..n*2).step_by(2) {
                            let (vr, vi) = (src[i], src[i + 1]);
                            let new_r = pr * vr - pi * vi;
                            let new_i = pr * vi + pi * vr;
                            pr = new_r;
                            pi = new_i;
                        }
                        *acc = pr;
                        *acc.add(1) = pi;
                    } else {
                        // Strided
                        let mut p = src_ptr;
                        for _ in 0..n {
                            let v = p as *const $T;
                            let (ar, ai, vr, vi) = (*acc, *acc.add(1), *v, *v.add(1));
                            *acc = ar * vr - ai * vi;
                            *acc.add(1) = ar * vi + ai * vr;
                            p = p.offset(stride);
                        }
                    }
                },
            );
        };
    }
    register_complex_loops!(reg, DTypeKind::Complex128, f64);
    register_complex_loops!(reg, DTypeKind::Complex64, f32);

    // ========================================================================
    // Unary loops
    // ========================================================================

    // Macro for strided unary loops with contiguous fast path
    macro_rules! register_strided_unary {
        ($reg:expr, $op:expr, $kind:expr, $T:ty, $f:expr) => {
            $reg.register_unary(
                $op,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    let (ss, so) = strides;
                    if ss == itemsize && so == itemsize {
                        // Contiguous fast path - LLVM can auto-vectorize
                        let src = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        let out = std::slice::from_raw_parts_mut(out_ptr as *mut $T, n);
                        for i in 0..n {
                            out[i] = $f(src[i]);
                        }
                    } else {
                        // Strided path
                        let mut sp = src_ptr;
                        let mut op = out_ptr;
                        for _ in 0..n {
                            let v = *(sp as *const $T);
                            *(op as *mut $T) = $f(v);
                            sp = sp.offset(ss);
                            op = op.offset(so);
                        }
                    }
                },
            );
        };
    }

    // All float unary ops (both f32 and f64)
    macro_rules! register_float_unary {
        ($reg:expr, $kind:expr, $T:ty) => {
            register_strided_unary!($reg, UnaryOp::Neg, $kind, $T, |v: $T| -v);
            register_strided_unary!($reg, UnaryOp::Abs, $kind, $T, |v: $T| v.abs());
            register_strided_unary!($reg, UnaryOp::Sqrt, $kind, $T, |v: $T| v.sqrt());
            register_strided_unary!($reg, UnaryOp::Floor, $kind, $T, |v: $T| v.floor());
            register_strided_unary!($reg, UnaryOp::Ceil, $kind, $T, |v: $T| v.ceil());
            register_strided_unary!($reg, UnaryOp::Exp, $kind, $T, |v: $T| v.exp());
            register_strided_unary!($reg, UnaryOp::Log, $kind, $T, |v: $T| v.ln());
            register_strided_unary!($reg, UnaryOp::Sin, $kind, $T, |v: $T| v.sin());
            register_strided_unary!($reg, UnaryOp::Cos, $kind, $T, |v: $T| v.cos());
            register_strided_unary!($reg, UnaryOp::Tan, $kind, $T, |v: $T| v.tan());
            register_strided_unary!($reg, UnaryOp::Sinh, $kind, $T, |v: $T| v.sinh());
            register_strided_unary!($reg, UnaryOp::Cosh, $kind, $T, |v: $T| v.cosh());
            register_strided_unary!($reg, UnaryOp::Tanh, $kind, $T, |v: $T| v.tanh());
            register_strided_unary!($reg, UnaryOp::Arcsin, $kind, $T, |v: $T| v.asin());
            register_strided_unary!($reg, UnaryOp::Arccos, $kind, $T, |v: $T| v.acos());
            register_strided_unary!($reg, UnaryOp::Arctan, $kind, $T, |v: $T| v.atan());

            register_strided_unary!($reg, UnaryOp::Square, $kind, $T, |v: $T| v * v);
            register_strided_unary!($reg, UnaryOp::Positive, $kind, $T, |v: $T| v);
            register_strided_unary!($reg, UnaryOp::Reciprocal, $kind, $T, |v: $T| 1.0 / v);
            register_strided_unary!($reg, UnaryOp::Exp2, $kind, $T, |v: $T| (2.0 as $T).powf(v));
            register_strided_unary!($reg, UnaryOp::Expm1, $kind, $T, |v: $T| v.exp_m1());
            register_strided_unary!($reg, UnaryOp::Log1p, $kind, $T, |v: $T| v.ln_1p());
            register_strided_unary!($reg, UnaryOp::Cbrt, $kind, $T, |v: $T| v.cbrt());
            register_strided_unary!($reg, UnaryOp::Trunc, $kind, $T, |v: $T| v.trunc());
            register_strided_unary!($reg, UnaryOp::Rint, $kind, $T, |v: $T| v.round());
            register_strided_unary!($reg, UnaryOp::Arcsinh, $kind, $T, |v: $T| v.asinh());
            register_strided_unary!($reg, UnaryOp::Arccosh, $kind, $T, |v: $T| v.acosh());
            register_strided_unary!($reg, UnaryOp::Arctanh, $kind, $T, |v: $T| v.atanh());
            register_strided_unary!($reg, UnaryOp::Signbit, $kind, $T, |v: $T| if v.is_sign_negative() { 1.0 } else { 0.0 });
        };
    }
    register_float_unary!(reg, DTypeKind::Float64, f64);
    register_float_unary!(reg, DTypeKind::Float32, f32);

    // Float16 unary loops (convert to f32 for ops)
    macro_rules! register_f16_unary {
        ($reg:expr, $op:expr, $f:expr) => {
            $reg.register_unary(
                $op,
                TypeSignature::unary(DTypeKind::Float16, DTypeKind::Float16),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let (ss, so) = strides;
                    let mut sp = src_ptr;
                    let mut op = out_ptr;
                    for _ in 0..n {
                        let v = (*(sp as *const f16)).to_f32();
                        *(op as *mut f16) = f16::from_f32($f(v));
                        sp = sp.offset(ss);
                        op = op.offset(so);
                    }
                },
            );
        };
    }
    register_f16_unary!(reg, UnaryOp::Neg, |v: f32| -v);
    register_f16_unary!(reg, UnaryOp::Abs, |v: f32| v.abs());
    register_f16_unary!(reg, UnaryOp::Sqrt, |v: f32| v.sqrt());
    register_f16_unary!(reg, UnaryOp::Floor, |v: f32| v.floor());
    register_f16_unary!(reg, UnaryOp::Ceil, |v: f32| v.ceil());
    register_f16_unary!(reg, UnaryOp::Exp, |v: f32| v.exp());
    register_f16_unary!(reg, UnaryOp::Log, |v: f32| v.ln());
    register_f16_unary!(reg, UnaryOp::Sin, |v: f32| v.sin());
    register_f16_unary!(reg, UnaryOp::Cos, |v: f32| v.cos());
    register_f16_unary!(reg, UnaryOp::Tan, |v: f32| v.tan());
    register_f16_unary!(reg, UnaryOp::Sinh, |v: f32| v.sinh());
    register_f16_unary!(reg, UnaryOp::Cosh, |v: f32| v.cosh());
    register_f16_unary!(reg, UnaryOp::Tanh, |v: f32| v.tanh());
    register_f16_unary!(reg, UnaryOp::Arcsin, |v: f32| v.asin());
    register_f16_unary!(reg, UnaryOp::Arccos, |v: f32| v.acos());
    register_f16_unary!(reg, UnaryOp::Arctan, |v: f32| v.atan());

    register_f16_unary!(reg, UnaryOp::Square, |v: f32| v * v);
    register_f16_unary!(reg, UnaryOp::Positive, |v: f32| v);
    register_f16_unary!(reg, UnaryOp::Reciprocal, |v: f32| 1.0 / v);
    register_f16_unary!(reg, UnaryOp::Exp2, |v: f32| 2.0f32.powf(v));
    register_f16_unary!(reg, UnaryOp::Expm1, |v: f32| v.exp_m1());
    register_f16_unary!(reg, UnaryOp::Log1p, |v: f32| v.ln_1p());
    register_f16_unary!(reg, UnaryOp::Cbrt, |v: f32| v.cbrt());
    register_f16_unary!(reg, UnaryOp::Trunc, |v: f32| v.trunc());
    register_f16_unary!(reg, UnaryOp::Rint, |v: f32| v.round());
    register_f16_unary!(reg, UnaryOp::Arcsinh, |v: f32| v.asinh());
    register_f16_unary!(reg, UnaryOp::Arccosh, |v: f32| v.acosh());
    register_f16_unary!(reg, UnaryOp::Arctanh, |v: f32| v.atanh());
    register_f16_unary!(reg, UnaryOp::Signbit, |v: f32| if v.is_sign_negative() { 1.0 } else { 0.0 });

    // Signed integer unary ops
    macro_rules! register_signed_int_unary {
        ($reg:expr, $kind:expr, $T:ty) => {
            register_strided_unary!($reg, UnaryOp::Neg, $kind, $T, |v: $T| -v);
            register_strided_unary!($reg, UnaryOp::Abs, $kind, $T, |v: $T| v.abs());
        };
    }
    register_signed_int_unary!(reg, DTypeKind::Int64, i64);
    register_signed_int_unary!(reg, DTypeKind::Int32, i32);
    register_signed_int_unary!(reg, DTypeKind::Int16, i16);

    // Unsigned integer abs (identity)
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Uint64, u64, |v: u64| v);
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Uint32, u32, |v: u32| v);
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Uint16, u16, |v: u16| v);
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Uint8, u8, |v: u8| v);

    // ========================================================================
    // Reduce loops
    // ========================================================================

    // Float reduce ops (use += and *= directly)
    macro_rules! register_float_reduce {
        ($reg:expr, $kind:expr, $T:ty) => {
            $reg.register_reduce(
                ReduceOp::Sum, TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, idx| unsafe { *(out_ptr as *mut $T).add(idx) = 0.0; },
                |acc_ptr, idx, val_ptr, byte_offset| unsafe {
                    let acc = (acc_ptr as *mut $T).add(idx);
                    *acc += *(val_ptr.offset(byte_offset) as *const $T);
                },
            );
            $reg.register_reduce(
                ReduceOp::Prod, TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, idx| unsafe { *(out_ptr as *mut $T).add(idx) = 1.0; },
                |acc_ptr, idx, val_ptr, byte_offset| unsafe {
                    let acc = (acc_ptr as *mut $T).add(idx);
                    *acc *= *(val_ptr.offset(byte_offset) as *const $T);
                },
            );
            $reg.register_reduce(
                ReduceOp::Max, TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, idx| unsafe { *(out_ptr as *mut $T).add(idx) = <$T>::NEG_INFINITY; },
                |acc_ptr, idx, val_ptr, byte_offset| unsafe {
                    let acc = (acc_ptr as *mut $T).add(idx);
                    let v = *(val_ptr.offset(byte_offset) as *const $T);
                    if v > *acc { *acc = v; }
                },
            );
            $reg.register_reduce(
                ReduceOp::Min, TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, idx| unsafe { *(out_ptr as *mut $T).add(idx) = <$T>::INFINITY; },
                |acc_ptr, idx, val_ptr, byte_offset| unsafe {
                    let acc = (acc_ptr as *mut $T).add(idx);
                    let v = *(val_ptr.offset(byte_offset) as *const $T);
                    if v < *acc { *acc = v; }
                },
            );
        };
    }
    register_float_reduce!(reg, DTypeKind::Float64, f64);
    register_float_reduce!(reg, DTypeKind::Float32, f32);

    // Float16 reduce loops (convert to f32 for ops)
    reg.register_reduce(
        ReduceOp::Sum, TypeSignature::reduce(DTypeKind::Float16, DTypeKind::Float16),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f16).add(idx) = f16::ZERO; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f16).add(idx);
            let a = (*acc).to_f32();
            let v = (*(val_ptr.offset(byte_offset) as *const f16)).to_f32();
            *acc = f16::from_f32(a + v);
        },
    );
    reg.register_reduce(
        ReduceOp::Prod, TypeSignature::reduce(DTypeKind::Float16, DTypeKind::Float16),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f16).add(idx) = f16::ONE; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f16).add(idx);
            let a = (*acc).to_f32();
            let v = (*(val_ptr.offset(byte_offset) as *const f16)).to_f32();
            *acc = f16::from_f32(a * v);
        },
    );
    reg.register_reduce(
        ReduceOp::Max, TypeSignature::reduce(DTypeKind::Float16, DTypeKind::Float16),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f16).add(idx) = f16::NEG_INFINITY; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f16).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f16);
            if v > *acc { *acc = v; }
        },
    );
    reg.register_reduce(
        ReduceOp::Min, TypeSignature::reduce(DTypeKind::Float16, DTypeKind::Float16),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f16).add(idx) = f16::INFINITY; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f16).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f16);
            if v < *acc { *acc = v; }
        },
    );

    // Integer reduce ops (use wrapping arithmetic)
    macro_rules! register_int_reduce {
        ($reg:expr, $kind:expr, $T:ty) => {
            $reg.register_reduce(
                ReduceOp::Sum, TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, idx| unsafe { *(out_ptr as *mut $T).add(idx) = 0; },
                |acc_ptr, idx, val_ptr, byte_offset| unsafe {
                    let acc = (acc_ptr as *mut $T).add(idx);
                    *acc = acc.read().wrapping_add(*(val_ptr.offset(byte_offset) as *const $T));
                },
            );
            $reg.register_reduce(
                ReduceOp::Prod, TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, idx| unsafe { *(out_ptr as *mut $T).add(idx) = 1; },
                |acc_ptr, idx, val_ptr, byte_offset| unsafe {
                    let acc = (acc_ptr as *mut $T).add(idx);
                    *acc = acc.read().wrapping_mul(*(val_ptr.offset(byte_offset) as *const $T));
                },
            );
            $reg.register_reduce(
                ReduceOp::Max, TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, idx| unsafe { *(out_ptr as *mut $T).add(idx) = <$T>::MIN; },
                |acc_ptr, idx, val_ptr, byte_offset| unsafe {
                    let acc = (acc_ptr as *mut $T).add(idx);
                    let v = *(val_ptr.offset(byte_offset) as *const $T);
                    if v > *acc { *acc = v; }
                },
            );
            $reg.register_reduce(
                ReduceOp::Min, TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, idx| unsafe { *(out_ptr as *mut $T).add(idx) = <$T>::MAX; },
                |acc_ptr, idx, val_ptr, byte_offset| unsafe {
                    let acc = (acc_ptr as *mut $T).add(idx);
                    let v = *(val_ptr.offset(byte_offset) as *const $T);
                    if v < *acc { *acc = v; }
                },
            );
        };
    }
    register_int_reduce!(reg, DTypeKind::Int64, i64);
    register_int_reduce!(reg, DTypeKind::Int32, i32);
    register_int_reduce!(reg, DTypeKind::Int16, i16);
    register_int_reduce!(reg, DTypeKind::Uint64, u64);
    register_int_reduce!(reg, DTypeKind::Uint32, u32);
    register_int_reduce!(reg, DTypeKind::Uint16, u16);
    register_int_reduce!(reg, DTypeKind::Uint8, u8);

    // bool reduce loops (Sum=any, Prod=all)
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

    // ========================================================================
    // Strided reduce loops (for axis reductions)
    // ========================================================================
    //
    // These process N elements at once, with contiguous fast path for SIMD.

    macro_rules! register_strided_reduce_float {
        ($reg:expr, $kind:expr, $T:ty) => {
            // Sum - uses multiple accumulators to break dependency chain for SIMD
            $reg.register_reduce_strided(
                ReduceOp::Sum,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, _idx| unsafe { *(out_ptr as *mut $T) = 0.0; },
                |acc_ptr, src_ptr, n, stride| unsafe {
                    let acc = acc_ptr as *mut $T;
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    if stride == itemsize {
                        // Contiguous: 8 accumulators break dependency chain for SIMD
                        let slice = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        let (mut s0, mut s1, mut s2, mut s3) = (0.0 as $T, 0.0 as $T, 0.0 as $T, 0.0 as $T);
                        let (mut s4, mut s5, mut s6, mut s7) = (0.0 as $T, 0.0 as $T, 0.0 as $T, 0.0 as $T);
                        let chunks = slice.chunks_exact(8);
                        let remainder = chunks.remainder();
                        for chunk in chunks {
                            s0 += chunk[0]; s1 += chunk[1]; s2 += chunk[2]; s3 += chunk[3];
                            s4 += chunk[4]; s5 += chunk[5]; s6 += chunk[6]; s7 += chunk[7];
                        }
                        *acc = (s0 + s1 + s2 + s3) + (s4 + s5 + s6 + s7) + remainder.iter().copied().sum::<$T>();
                    } else {
                        // Strided: simple loop (no SIMD benefit)
                        let mut ptr = src_ptr;
                        let mut sum: $T = 0.0;
                        for _ in 0..n {
                            sum += *(ptr as *const $T);
                            ptr = ptr.offset(stride);
                        }
                        *acc = sum;
                    }
                },
            );
            // Prod
            $reg.register_reduce_strided(
                ReduceOp::Prod,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, _idx| unsafe { *(out_ptr as *mut $T) = 1.0; },
                |acc_ptr, src_ptr, n, stride| unsafe {
                    let acc = acc_ptr as *mut $T;
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    if stride == itemsize {
                        let slice = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        *acc = slice.iter().product();
                    } else {
                        let mut ptr = src_ptr;
                        let mut prod: $T = 1.0;
                        for _ in 0..n {
                            prod *= *(ptr as *const $T);
                            ptr = ptr.offset(stride);
                        }
                        *acc = prod;
                    }
                },
            );
            // Max - uses multiple accumulators to break dependency chain
            $reg.register_reduce_strided(
                ReduceOp::Max,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, _idx| unsafe { *(out_ptr as *mut $T) = <$T>::NEG_INFINITY; },
                |acc_ptr, src_ptr, n, stride| unsafe {
                    let acc = acc_ptr as *mut $T;
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    if stride == itemsize {
                        let slice = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        let (mut m0, mut m1, mut m2, mut m3) = (<$T>::NEG_INFINITY, <$T>::NEG_INFINITY, <$T>::NEG_INFINITY, <$T>::NEG_INFINITY);
                        let chunks = slice.chunks_exact(4);
                        let remainder = chunks.remainder();
                        for chunk in chunks {
                            m0 = m0.max(chunk[0]); m1 = m1.max(chunk[1]);
                            m2 = m2.max(chunk[2]); m3 = m3.max(chunk[3]);
                        }
                        let mut result = m0.max(m1).max(m2.max(m3));
                        for &v in remainder { result = result.max(v); }
                        *acc = result;
                    } else {
                        let mut ptr = src_ptr;
                        let mut max_val: $T = <$T>::NEG_INFINITY;
                        for _ in 0..n {
                            let v = *(ptr as *const $T);
                            max_val = max_val.max(v);
                            ptr = ptr.offset(stride);
                        }
                        *acc = max_val;
                    }
                },
            );
            // Min - uses multiple accumulators to break dependency chain
            $reg.register_reduce_strided(
                ReduceOp::Min,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, _idx| unsafe { *(out_ptr as *mut $T) = <$T>::INFINITY; },
                |acc_ptr, src_ptr, n, stride| unsafe {
                    let acc = acc_ptr as *mut $T;
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    if stride == itemsize {
                        let slice = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        let (mut m0, mut m1, mut m2, mut m3) = (<$T>::INFINITY, <$T>::INFINITY, <$T>::INFINITY, <$T>::INFINITY);
                        let chunks = slice.chunks_exact(4);
                        let remainder = chunks.remainder();
                        for chunk in chunks {
                            m0 = m0.min(chunk[0]); m1 = m1.min(chunk[1]);
                            m2 = m2.min(chunk[2]); m3 = m3.min(chunk[3]);
                        }
                        let mut result = m0.min(m1).min(m2.min(m3));
                        for &v in remainder { result = result.min(v); }
                        *acc = result;
                    } else {
                        let mut ptr = src_ptr;
                        let mut min_val: $T = <$T>::INFINITY;
                        for _ in 0..n {
                            let v = *(ptr as *const $T);
                            min_val = min_val.min(v);
                            ptr = ptr.offset(stride);
                        }
                        *acc = min_val;
                    }
                },
            );
        };
    }
    register_strided_reduce_float!(reg, DTypeKind::Float64, f64);
    register_strided_reduce_float!(reg, DTypeKind::Float32, f32);

    // Integer strided reduce loops
    macro_rules! register_strided_reduce_int {
        ($reg:expr, $kind:expr, $T:ty) => {
            // Sum (wrapping)
            $reg.register_reduce_strided(
                ReduceOp::Sum,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, _idx| unsafe { *(out_ptr as *mut $T) = 0; },
                |acc_ptr, src_ptr, n, stride| unsafe {
                    let acc = acc_ptr as *mut $T;
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    if stride == itemsize {
                        let slice = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        *acc = slice.iter().fold(0 as $T, |a, &b| a.wrapping_add(b));
                    } else {
                        let mut ptr = src_ptr;
                        let mut sum: $T = 0;
                        for _ in 0..n {
                            sum = sum.wrapping_add(*(ptr as *const $T));
                            ptr = ptr.offset(stride);
                        }
                        *acc = sum;
                    }
                },
            );
            // Prod (wrapping)
            $reg.register_reduce_strided(
                ReduceOp::Prod,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, _idx| unsafe { *(out_ptr as *mut $T) = 1; },
                |acc_ptr, src_ptr, n, stride| unsafe {
                    let acc = acc_ptr as *mut $T;
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    if stride == itemsize {
                        let slice = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        *acc = slice.iter().fold(1 as $T, |a, &b| a.wrapping_mul(b));
                    } else {
                        let mut ptr = src_ptr;
                        let mut prod: $T = 1;
                        for _ in 0..n {
                            prod = prod.wrapping_mul(*(ptr as *const $T));
                            ptr = ptr.offset(stride);
                        }
                        *acc = prod;
                    }
                },
            );
            // Max
            $reg.register_reduce_strided(
                ReduceOp::Max,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, _idx| unsafe { *(out_ptr as *mut $T) = <$T>::MIN; },
                |acc_ptr, src_ptr, n, stride| unsafe {
                    let acc = acc_ptr as *mut $T;
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    if stride == itemsize {
                        let slice = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        *acc = slice.iter().cloned().max().unwrap_or(<$T>::MIN);
                    } else {
                        let mut ptr = src_ptr;
                        let mut max_val: $T = <$T>::MIN;
                        for _ in 0..n {
                            let v = *(ptr as *const $T);
                            if v > max_val { max_val = v; }
                            ptr = ptr.offset(stride);
                        }
                        *acc = max_val;
                    }
                },
            );
            // Min
            $reg.register_reduce_strided(
                ReduceOp::Min,
                TypeSignature::reduce($kind.clone(), $kind.clone()),
                |out_ptr, _idx| unsafe { *(out_ptr as *mut $T) = <$T>::MAX; },
                |acc_ptr, src_ptr, n, stride| unsafe {
                    let acc = acc_ptr as *mut $T;
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    if stride == itemsize {
                        let slice = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        *acc = slice.iter().cloned().min().unwrap_or(<$T>::MAX);
                    } else {
                        let mut ptr = src_ptr;
                        let mut min_val: $T = <$T>::MAX;
                        for _ in 0..n {
                            let v = *(ptr as *const $T);
                            if v < min_val { min_val = v; }
                            ptr = ptr.offset(stride);
                        }
                        *acc = min_val;
                    }
                },
            );
        };
    }
    register_strided_reduce_int!(reg, DTypeKind::Int64, i64);
    register_strided_reduce_int!(reg, DTypeKind::Int32, i32);
    register_strided_reduce_int!(reg, DTypeKind::Int16, i16);
    register_strided_reduce_int!(reg, DTypeKind::Uint64, u64);
    register_strided_reduce_int!(reg, DTypeKind::Uint32, u32);
    register_strided_reduce_int!(reg, DTypeKind::Uint16, u16);
    register_strided_reduce_int!(reg, DTypeKind::Uint8, u8);

    // ========================================================================
    // Bitwise operations (integer/bool types)
    // ========================================================================

    // Macro for bitwise binary loops with contiguous fast path
    macro_rules! register_bitwise_binary {
        ($reg:expr, $op:expr, $kind:expr, $T:ty, $f:expr) => {
            $reg.register_bitwise_binary(
                $op,
                $kind.clone(),
                |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    let (sa, sb, so) = strides;
                    if sa == itemsize && sb == itemsize && so == itemsize {
                        // Contiguous fast path - LLVM can auto-vectorize
                        let a = std::slice::from_raw_parts(a_ptr as *const $T, n);
                        let b = std::slice::from_raw_parts(b_ptr as *const $T, n);
                        let out = std::slice::from_raw_parts_mut(out_ptr as *mut $T, n);
                        for i in 0..n {
                            out[i] = $f(a[i], b[i]);
                        }
                    } else {
                        // Strided path
                        let mut ap = a_ptr;
                        let mut bp = b_ptr;
                        let mut op = out_ptr;
                        for _ in 0..n {
                            let a = *(ap as *const $T);
                            let b = *(bp as *const $T);
                            *(op as *mut $T) = $f(a, b);
                            ap = ap.offset(sa);
                            bp = bp.offset(sb);
                            op = op.offset(so);
                        }
                    }
                },
            );
        };
    }

    // Macro for bitwise NOT loops
    macro_rules! register_bitwise_not {
        ($reg:expr, $kind:expr, $T:ty) => {
            $reg.register_bitwise_not(
                $kind.clone(),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    let (ss, so) = strides;
                    if ss == itemsize && so == itemsize {
                        // Contiguous fast path
                        let src = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        let out = std::slice::from_raw_parts_mut(out_ptr as *mut $T, n);
                        for i in 0..n {
                            out[i] = !src[i];
                        }
                    } else {
                        // Strided path
                        let mut sp = src_ptr;
                        let mut op = out_ptr;
                        for _ in 0..n {
                            *(op as *mut $T) = !(*(sp as *const $T));
                            sp = sp.offset(ss);
                            op = op.offset(so);
                        }
                    }
                },
            );
        };
    }

    // Register all bitwise ops for an integer type
    macro_rules! register_bitwise_ops {
        ($reg:expr, $kind:expr, $T:ty) => {
            register_bitwise_binary!($reg, BitwiseOp::And, $kind, $T, |a: $T, b: $T| a & b);
            register_bitwise_binary!($reg, BitwiseOp::Or, $kind, $T, |a: $T, b: $T| a | b);
            register_bitwise_binary!($reg, BitwiseOp::Xor, $kind, $T, |a: $T, b: $T| a ^ b);
            register_bitwise_binary!($reg, BitwiseOp::LeftShift, $kind, $T, |a: $T, b: $T| a.wrapping_shl(b as u32));
            register_bitwise_binary!($reg, BitwiseOp::RightShift, $kind, $T, |a: $T, b: $T| a.wrapping_shr(b as u32));
            register_bitwise_not!($reg, $kind, $T);
        };
    }

    // Register for all integer types
    register_bitwise_ops!(reg, DTypeKind::Int64, i64);
    register_bitwise_ops!(reg, DTypeKind::Int32, i32);
    register_bitwise_ops!(reg, DTypeKind::Int16, i16);
    register_bitwise_ops!(reg, DTypeKind::Uint64, u64);
    register_bitwise_ops!(reg, DTypeKind::Uint32, u32);
    register_bitwise_ops!(reg, DTypeKind::Uint16, u16);
    register_bitwise_ops!(reg, DTypeKind::Uint8, u8);

    // Bool bitwise (use logical operators)
    reg.register_bitwise_binary(
        BitwiseOp::And,
        DTypeKind::Bool,
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            if sa == 1 && sb == 1 && so == 1 {
                let a = std::slice::from_raw_parts(a_ptr, n);
                let b = std::slice::from_raw_parts(b_ptr, n);
                let out = std::slice::from_raw_parts_mut(out_ptr, n);
                for i in 0..n {
                    out[i] = (a[i] != 0 && b[i] != 0) as u8;
                }
            } else {
                let (mut ap, mut bp, mut op) = (a_ptr, b_ptr, out_ptr);
                for _ in 0..n {
                    *op = (*ap != 0 && *bp != 0) as u8;
                    ap = ap.offset(sa);
                    bp = bp.offset(sb);
                    op = op.offset(so);
                }
            }
        },
    );
    reg.register_bitwise_binary(
        BitwiseOp::Or,
        DTypeKind::Bool,
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            if sa == 1 && sb == 1 && so == 1 {
                let a = std::slice::from_raw_parts(a_ptr, n);
                let b = std::slice::from_raw_parts(b_ptr, n);
                let out = std::slice::from_raw_parts_mut(out_ptr, n);
                for i in 0..n {
                    out[i] = (a[i] != 0 || b[i] != 0) as u8;
                }
            } else {
                let (mut ap, mut bp, mut op) = (a_ptr, b_ptr, out_ptr);
                for _ in 0..n {
                    *op = (*ap != 0 || *bp != 0) as u8;
                    ap = ap.offset(sa);
                    bp = bp.offset(sb);
                    op = op.offset(so);
                }
            }
        },
    );
    reg.register_bitwise_binary(
        BitwiseOp::Xor,
        DTypeKind::Bool,
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            if sa == 1 && sb == 1 && so == 1 {
                let a = std::slice::from_raw_parts(a_ptr, n);
                let b = std::slice::from_raw_parts(b_ptr, n);
                let out = std::slice::from_raw_parts_mut(out_ptr, n);
                for i in 0..n {
                    out[i] = ((a[i] != 0) != (b[i] != 0)) as u8;
                }
            } else {
                let (mut ap, mut bp, mut op) = (a_ptr, b_ptr, out_ptr);
                for _ in 0..n {
                    *op = ((*ap != 0) != (*bp != 0)) as u8;
                    ap = ap.offset(sa);
                    bp = bp.offset(sb);
                    op = op.offset(so);
                }
            }
        },
    );
    reg.register_bitwise_not(
        DTypeKind::Bool,
        |src_ptr, out_ptr, n, strides| unsafe {
            let (ss, so) = strides;
            if ss == 1 && so == 1 {
                let src = std::slice::from_raw_parts(src_ptr, n);
                let out = std::slice::from_raw_parts_mut(out_ptr, n);
                for i in 0..n {
                    out[i] = (src[i] == 0) as u8;
                }
            } else {
                let (mut sp, mut op) = (src_ptr, out_ptr);
                for _ in 0..n {
                    *op = (*sp == 0) as u8;
                    sp = sp.offset(ss);
                    op = op.offset(so);
                }
            }
        },
    );

    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_signature_equality() {
        let sig1 = TypeSignature::binary(DTypeKind::Float64, DTypeKind::Float64, DTypeKind::Float64);
        let sig2 = TypeSignature::binary(DTypeKind::Float64, DTypeKind::Float64, DTypeKind::Float64);
        let sig3 = TypeSignature::binary(DTypeKind::Float32, DTypeKind::Float32, DTypeKind::Float32);

        assert_eq!(sig1, sig2);
        assert_ne!(sig1, sig3);
    }

    #[test]
    fn test_registry_lookup_found() {
        let reg = init_default_loops();
        let result = reg.lookup_binary(BinaryOp::Add, DTypeKind::Float64, DTypeKind::Float64);
        assert!(result.is_some());
        let (_, out_dtype) = result.unwrap();
        assert_eq!(out_dtype, DTypeKind::Float64);
    }

    #[test]
    fn test_registry_lookup_not_found() {
        let reg = init_default_loops();
        // Mixed types not registered
        let result = reg.lookup_binary(BinaryOp::Add, DTypeKind::Float64, DTypeKind::Float32);
        assert!(result.is_none());
    }

    #[test]
    fn test_f64_add_loop() {
        let reg = init_default_loops();
        let (loop_fn, _) = reg
            .lookup_binary(BinaryOp::Add, DTypeKind::Float64, DTypeKind::Float64)
            .unwrap();

        let a: [f64; 2] = [3.0, 10.0];
        let b: [f64; 2] = [4.0, 20.0];
        let mut out: [f64; 2] = [0.0, 0.0];

        unsafe {
            loop_fn(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                2,
                (8, 8, 8),  // f64 strides
            );
        }

        assert_eq!(out, [7.0, 30.0]);
    }

    #[test]
    fn test_i64_mul_loop() {
        let reg = init_default_loops();
        let (loop_fn, _) = reg
            .lookup_binary(BinaryOp::Mul, DTypeKind::Int64, DTypeKind::Int64)
            .unwrap();

        let a: [i64; 2] = [5, 7];
        let b: [i64; 2] = [6, 8];
        let mut out: [i64; 2] = [0, 0];

        unsafe {
            loop_fn(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                2,
                (8, 8, 8),  // i64 strides
            );
        }

        assert_eq!(out, [30, 56]);
    }

    #[test]
    fn test_global_registry() {
        let reg = registry().read().unwrap();
        let result = reg.lookup_binary(BinaryOp::Add, DTypeKind::Float64, DTypeKind::Float64);
        assert!(result.is_some());
    }

    #[test]
    fn test_f64_neg_loop() {
        let reg = init_default_loops();
        let (loop_fn, _) = reg
            .lookup_unary(UnaryOp::Neg, DTypeKind::Float64)
            .unwrap();

        let src: [f64; 2] = [3.5, -2.0];
        let mut out: [f64; 2] = [0.0, 0.0];

        unsafe {
            loop_fn(
                src.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                2,
                (8, 8),  // f64 strides
            );
        }

        assert_eq!(out, [-3.5, 2.0]);
    }

    #[test]
    fn test_f64_sqrt_loop() {
        let reg = init_default_loops();
        let (loop_fn, _) = reg
            .lookup_unary(UnaryOp::Sqrt, DTypeKind::Float64)
            .unwrap();

        let src: [f64; 2] = [16.0, 25.0];
        let mut out: [f64; 2] = [0.0, 0.0];

        unsafe {
            loop_fn(
                src.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                2,
                (8, 8),  // f64 strides
            );
        }

        assert_eq!(out, [4.0, 5.0]);
    }

    #[test]
    fn test_unary_lookup_not_found() {
        let reg = init_default_loops();
        // Exp not registered for i32
        let result = reg.lookup_unary(UnaryOp::Exp, DTypeKind::Int32);
        assert!(result.is_none());
    }

    #[test]
    fn test_reduce_lookup_found() {
        let reg = init_default_loops();
        let result = reg.lookup_reduce(ReduceOp::Sum, DTypeKind::Float64);
        assert!(result.is_some());
        let (_, _, out_dtype) = result.unwrap();
        assert_eq!(out_dtype, DTypeKind::Float64);
    }

    #[test]
    fn test_f64_sum_loop() {
        let reg = init_default_loops();
        let (init_fn, acc_fn, _) = reg
            .lookup_reduce(ReduceOp::Sum, DTypeKind::Float64)
            .unwrap();

        let values: [f64; 3] = [1.0, 2.0, 3.0];
        let mut acc: f64 = 0.0;

        unsafe {
            init_fn(&mut acc as *mut f64 as *mut u8, 0);
            for v in &values {
                acc_fn(
                    &mut acc as *mut f64 as *mut u8,
                    0,
                    v as *const f64 as *const u8,
                    0,
                );
            }
        }

        assert_eq!(acc, 6.0);
    }

    #[test]
    fn test_f64_max_loop() {
        let reg = init_default_loops();
        let (init_fn, acc_fn, _) = reg
            .lookup_reduce(ReduceOp::Max, DTypeKind::Float64)
            .unwrap();

        let values: [f64; 4] = [1.0, 5.0, 2.0, 3.0];
        let mut acc: f64 = 0.0;

        unsafe {
            init_fn(&mut acc as *mut f64 as *mut u8, 0);
            for v in &values {
                acc_fn(
                    &mut acc as *mut f64 as *mut u8,
                    0,
                    v as *const f64 as *const u8,
                    0,
                );
            }
        }

        assert_eq!(acc, 5.0);
    }

    #[test]
    fn test_i64_prod_loop() {
        let reg = init_default_loops();
        let (init_fn, acc_fn, _) = reg
            .lookup_reduce(ReduceOp::Prod, DTypeKind::Int64)
            .unwrap();

        let values: [i64; 4] = [2, 3, 4, 5];
        let mut acc: i64 = 0;

        unsafe {
            init_fn(&mut acc as *mut i64 as *mut u8, 0);
            for v in &values {
                acc_fn(
                    &mut acc as *mut i64 as *mut u8,
                    0,
                    v as *const i64 as *const u8,
                    0,
                );
            }
        }

        assert_eq!(acc, 120);
    }
}
