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

        // Box-Muller generates pairs - use both values
        let pairs = size / 2;
        for _ in 0..pairs {
            let (z0, z1) = self.next_standard_normal_pair();
            data.push(z0);
            data.push(z1);
        }
        // Handle odd size
        if size % 2 == 1 {
            data.push(self.next_standard_normal());
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate a pair of standard normal values using Box-Muller transform.
    #[inline]
    fn next_standard_normal_pair(&mut self) -> (f64, f64) {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 0.0 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                return (r * theta.cos(), r * theta.sin());
            }
        }
    }

    /// Generate one standard normal value using Box-Muller transform.
    pub fn next_standard_normal(&mut self) -> f64 {
        self.next_standard_normal_pair().0
    }

    /// Generate samples from normal distribution.
    pub fn normal(&mut self, loc: f64, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        // Box-Muller generates pairs - use both values
        let pairs = size / 2;
        for _ in 0..pairs {
            let (z0, z1) = self.next_standard_normal_pair();
            data.push(loc + scale * z0);
            data.push(loc + scale * z1);
        }
        if size % 2 == 1 {
            data.push(loc + scale * self.next_standard_normal());
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

    /// Random permutation of integers from 0 to n-1, or a shuffled copy of an array.
    pub fn permutation(&mut self, n: usize) -> RumpyArray {
        let mut data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = self.bounded_uint64((i + 1) as u64) as usize;
            data.swap(i, j);
        }
        RumpyArray::from_vec(data, DType::float64())
    }

    /// Shuffle data in place using Fisher-Yates algorithm.
    /// Takes a mutable slice of f64 values.
    pub fn shuffle_data(&mut self, data: &mut [f64]) {
        let n = data.len();
        for i in (1..n).rev() {
            let j = self.bounded_uint64((i + 1) as u64) as usize;
            data.swap(i, j);
        }
    }

    /// Generate one standard gamma value using Marsaglia and Tsang's method.
    fn next_standard_gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            // For shape < 1, use: Gamma(a) = Gamma(a+1) * U^(1/a)
            let u = self.next_f64();
            return self.next_standard_gamma(shape + 1.0) * u.powf(1.0 / shape);
        }

        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = self.next_standard_normal();
            let v = 1.0 + c * x;
            if v > 0.0 {
                let v = v * v * v;
                let u = self.next_f64();
                let x2 = x * x;
                if u < 1.0 - 0.0331 * x2 * x2 || u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
                    return d * v;
                }
            }
        }
    }

    /// Generate samples from gamma distribution.
    pub fn gamma(&mut self, shape_param: f64, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(scale * self.next_standard_gamma(shape_param));
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from beta distribution.
    /// Beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b))
    pub fn beta(&mut self, a: f64, b: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            let x = self.next_standard_gamma(a);
            let y = self.next_standard_gamma(b);
            data.push(x / (x + y));
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate one Poisson sample using Knuth's algorithm for small lambda,
    /// or the transformed rejection method for large lambda.
    fn next_poisson(&mut self, lam: f64) -> u64 {
        if lam < 30.0 {
            // Knuth's algorithm for small lambda
            let l = (-lam).exp();
            let mut k = 0u64;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.next_f64();
                if p <= l {
                    return k - 1;
                }
            }
        } else {
            // For large lambda, use normal approximation with correction
            loop {
                let x = self.next_standard_normal() * lam.sqrt() + lam;
                if x >= 0.0 {
                    return x.round() as u64;
                }
            }
        }
    }

    /// Generate samples from Poisson distribution.
    pub fn poisson(&mut self, lam: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(self.next_poisson(lam) as f64);
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::int64())
    }

    /// Generate one binomial sample.
    fn next_binomial(&mut self, n: u64, p: f64) -> u64 {
        if n == 0 || p == 0.0 {
            return 0;
        }
        if p == 1.0 {
            return n;
        }

        // For small n, use direct simulation
        if n < 25 {
            let mut successes = 0u64;
            for _ in 0..n {
                if self.next_f64() < p {
                    successes += 1;
                }
            }
            return successes;
        }

        // For large n, use normal approximation
        let mean = n as f64 * p;
        let std = (mean * (1.0 - p)).sqrt();
        loop {
            let x = self.next_standard_normal() * std + mean + 0.5;
            if x >= 0.0 && x <= n as f64 {
                return x.floor() as u64;
            }
        }
    }

    /// Generate samples from binomial distribution.
    pub fn binomial(&mut self, n: u64, p: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(self.next_binomial(n, p) as f64);
        }

        RumpyArray::from_vec_with_shape(data, shape, DType::int64())
    }

    /// Generate samples from chi-square distribution.
    /// Chi-square(k) = Gamma(k/2, 2)
    pub fn chisquare(&mut self, df: f64, shape: Vec<usize>) -> RumpyArray {
        self.gamma(df / 2.0, 2.0, shape)
    }

    /// Generate samples from multivariate normal distribution.
    /// Uses Cholesky decomposition: X = mean + L @ Z where L @ L^T = cov
    pub fn multivariate_normal(
        &mut self,
        mean: &[f64],
        cov: &[Vec<f64>],
        size: usize,
    ) -> RumpyArray {
        let d = mean.len();

        // Cholesky decomposition (lower triangular)
        let mut l = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..=i {
                let mut sum = cov[i][j];
                sum -= l[i][..j].iter().zip(&l[j][..j]).map(|(a, b)| a * b).sum::<f64>();
                if i == j {
                    l[i][j] = sum.sqrt();
                } else {
                    l[i][j] = sum / l[j][j];
                }
            }
        }

        // Generate samples
        let mut data = Vec::with_capacity(size * d);
        for _ in 0..size {
            // Generate standard normal vector
            let z: Vec<f64> = (0..d).map(|_| self.next_standard_normal()).collect();

            // Transform: x = mean + L @ z
            for i in 0..d {
                let mut val = mean[i];
                for j in 0..=i {
                    val += l[i][j] * z[j];
                }
                data.push(val);
            }
        }

        RumpyArray::from_vec_with_shape(data, vec![size, d], DType::float64())
    }
}
