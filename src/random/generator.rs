// Generator struct - main interface for random number generation

use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64Dxsm as RandPcg64Dxsm;

use crate::array::{DType, RumpyArray};
use super::pcg64::Pcg64Dxsm;

/// RNG backend - either rand_pcg's version or our numpy-compatible one.
enum RngBackend {
    RandPcg(RandPcg64Dxsm),
    NumpyCompat(Pcg64Dxsm),
}

/// Random number generator matching numpy.random.Generator API.
/// Uses PCG64DXSM BitGenerator for numpy compatibility.
pub struct Generator {
    rng: RngBackend,
}

impl Generator {
    /// Create a new Generator with given seed.
    /// Note: Uses SplitMix64 for seed expansion, which differs from numpy's SeedSequence.
    /// For exact numpy compatibility, use `from_numpy_state()` with state extracted from numpy.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RngBackend::RandPcg(RandPcg64Dxsm::seed_from_u64(seed)),
        }
    }

    /// Create Generator with explicit 128-bit state and increment from numpy.
    /// Use this for exact numpy compatibility by extracting state from numpy:
    /// ```python
    /// bg = np.random.PCG64DXSM(seed)
    /// state = bg.state['state']['state']
    /// inc = bg.state['state']['inc']
    /// ```
    pub fn from_numpy_state(state: u128, inc: u128) -> Self {
        Self {
            rng: RngBackend::NumpyCompat(Pcg64Dxsm::from_numpy_state(state, inc)),
        }
    }

    /// Generate next raw u64 value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        match &mut self.rng {
            RngBackend::RandPcg(rng) => rng.next_u64(),
            RngBackend::NumpyCompat(rng) => rng.next_u64(),
        }
    }

    /// Generate uniform random float64 in [0, 1).
    /// Uses 53-bit precision (IEEE 754 double mantissa).
    /// Algorithm: (u64 >> 11) * (1.0 / 2^53)
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// Generate array of uniform random floats in [0, 1).
    pub fn random(&mut self, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(self.next_f64());
        }
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate random integers in [low, high).
    pub fn integers(&mut self, low: i64, high: i64, shape: Vec<usize>) -> RumpyArray {
        let range = (high - low) as u64;
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            // Bounded rejection sampling for uniform distribution
            let val = self.bounded_uint64(range);
            data.push((low + val as i64) as f64);
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::int64())
    }

    /// Generate bounded random u64 in [0, range) using Lemire's algorithm.
    pub fn bounded_uint64(&mut self, range: u64) -> u64 {
        if range == 0 {
            return 0;
        }
        let mut x = self.next_u64();
        let mut m = (x as u128) * (range as u128);
        let mut l = m as u64;

        if l < range {
            let t = range.wrapping_neg() % range;
            while l < t {
                x = self.next_u64();
                m = (x as u128) * (range as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Generate uniform random floats in [low, high).
    pub fn uniform(&mut self, low: f64, high: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        let range = high - low;

        for _ in 0..size {
            data.push(low + self.next_f64() * range);
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from standard normal distribution using Box-Muller.
    pub fn standard_normal(&mut self, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(self.next_standard_normal());
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate one standard normal value using Box-Muller transform.
    pub fn next_standard_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 0.0 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                return r * theta.cos();
            }
        }
    }

    /// Generate samples from normal distribution.
    pub fn normal(&mut self, loc: f64, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            let z = self.next_standard_normal();
            data.push(loc + scale * z);
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from standard exponential distribution.
    pub fn standard_exponential(&mut self, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(self.next_standard_exponential());
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate one standard exponential value using inverse transform.
    pub fn next_standard_exponential(&mut self) -> f64 {
        loop {
            let u = self.next_f64();
            if u > 0.0 {
                return -u.ln();
            }
        }
    }

    /// Generate samples from exponential distribution.
    pub fn exponential(&mut self, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            let z = self.next_standard_exponential();
            data.push(scale * z);
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }
}
