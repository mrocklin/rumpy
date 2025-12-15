// Generator struct - main interface for random number generation
//
// Architecture: Generator implements RngCore, enabling use with rand_distr
// for optimized distribution sampling (Ziggurat for normal/exponential).

use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64Dxsm as RandPcg64Dxsm;
use rand_distr::Distribution;

use crate::array::{DType, RumpyArray};
use super::pcg64::Pcg64Dxsm;

/// RNG backend - either rand_pcg's version or our numpy-compatible one.
enum RngBackend {
    RandPcg(RandPcg64Dxsm),
    NumpyCompat(Pcg64Dxsm),
}

/// Random number generator matching numpy.random.Generator API.
/// Uses PCG64DXSM BitGenerator for numpy compatibility.
///
/// Implements `RngCore` for compatibility with `rand_distr` optimized distributions.
pub struct Generator {
    rng: RngBackend,
}

// Implement RngCore to enable use with rand_distr distributions
impl RngCore for Generator {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.raw_u64() as u32
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.raw_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        rand_core::impls::fill_bytes_via_next(self, dest)
    }
}

impl Generator {
    /// Generate next raw u64 value (internal method to avoid trait conflict).
    #[inline]
    fn raw_u64(&mut self) -> u64 {
        match &mut self.rng {
            RngBackend::RandPcg(rng) => rng.next_u64(),
            RngBackend::NumpyCompat(rng) => rng.next_u64(),
        }
    }

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
        let mut data = vec![0.0f64; size];
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        for d in data.iter_mut() {
            *d = (self.next_u64() >> 11) as f64 * SCALE;
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

    /// Generate samples from standard normal distribution using Ziggurat algorithm.
    pub fn standard_normal(&mut self, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::StandardNormal;
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate one standard normal value using Ziggurat algorithm.
    #[inline]
    pub fn next_standard_normal(&mut self) -> f64 {
        rand_distr::StandardNormal.sample(self)
    }

    /// Generate samples from normal distribution using Ziggurat algorithm.
    pub fn normal(&mut self, loc: f64, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Normal::new(loc, scale).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from standard exponential distribution using Ziggurat algorithm.
    pub fn standard_exponential(&mut self, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Exp1;
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate one standard exponential value using Ziggurat algorithm.
    #[inline]
    pub fn next_standard_exponential(&mut self) -> f64 {
        rand_distr::Exp1.sample(self)
    }

    /// Generate samples from exponential distribution.
    pub fn exponential(&mut self, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Exp::new(1.0 / scale).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
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

    /// Generate one standard gamma value using rand_distr (optimized algorithm).
    #[inline]
    fn next_standard_gamma(&mut self, shape_param: f64) -> f64 {
        rand_distr::Gamma::new(shape_param, 1.0).unwrap().sample(self)
    }

    /// Generate samples from gamma distribution.
    pub fn gamma(&mut self, shape_param: f64, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Gamma::new(shape_param, scale).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from beta distribution.
    pub fn beta(&mut self, a: f64, b: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Beta::new(a, b).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from Poisson distribution.
    pub fn poisson(&mut self, lam: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Poisson::new(lam).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self) as f64).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::int64())
    }

    /// Generate samples from binomial distribution.
    pub fn binomial(&mut self, n: u64, p: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Binomial::new(n, p).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self) as f64).collect();
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

    // === Tier 1 Distributions: Common ===

    /// Generate samples from lognormal distribution.
    pub fn lognormal(&mut self, mean: f64, sigma: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::LogNormal::new(mean, sigma).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from Laplace distribution (inverse CDF).
    pub fn laplace(&mut self, loc: f64, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| {
            let u = self.next_f64() - 0.5;
            loc - scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
        }).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from logistic distribution (inverse CDF).
    pub fn logistic(&mut self, loc: f64, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| {
            let u = self.next_f64();
            loc + scale * (u / (1.0 - u)).ln()
        }).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from Rayleigh distribution (inverse CDF).
    pub fn rayleigh(&mut self, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| {
            scale * (-2.0 * (1.0 - self.next_f64()).ln()).sqrt()
        }).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from Weibull distribution.
    pub fn weibull(&mut self, a: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Weibull::new(1.0, a).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    // === Tier 2 Distributions: Discrete ===

    /// Generate samples from geometric distribution.
    pub fn geometric(&mut self, p: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Geometric::new(p).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self) as f64).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::int64())
    }

    /// Generate samples from negative binomial distribution.
    pub fn negative_binomial(&mut self, n: f64, p: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        // Use gamma-Poisson mixture: NB(n, p) = Poisson(Gamma(n, (1-p)/p))
        let gamma_dist = rand_distr::Gamma::new(n, (1.0 - p) / p).unwrap();
        let data: Vec<f64> = (0..size).map(|_| {
            let y = gamma_dist.sample(self);
            rand_distr::Poisson::new(y).unwrap().sample(self) as f64
        }).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::int64())
    }

    /// Generate samples from hypergeometric distribution.
    pub fn hypergeometric(&mut self, ngood: u64, nbad: u64, nsample: u64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Hypergeometric::new(ngood + nbad, ngood, nsample).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self) as f64).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::int64())
    }

    /// Generate samples from multinomial distribution.
    pub fn multinomial(&mut self, n: u64, pvals: &[f64], size: usize) -> RumpyArray {
        let k = pvals.len();
        let mut data = vec![0.0; size * k];
        for i in 0..size {
            let row = &mut data[i * k..(i + 1) * k];
            let mut remaining = n;
            let mut p_remaining = 1.0;
            for j in 0..k - 1 {
                if p_remaining > 0.0 {
                    let p = pvals[j] / p_remaining;
                    let c = rand_distr::Binomial::new(remaining, p).unwrap().sample(self);
                    row[j] = c as f64;
                    remaining -= c;
                    p_remaining -= pvals[j];
                }
            }
            row[k - 1] = remaining as f64;
        }
        RumpyArray::from_vec_with_shape(data, vec![size, k], DType::float64())
    }

    /// Generate samples from Zipf distribution (rejection sampling).
    pub fn zipf(&mut self, a: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        // Use rejection sampling (rand_distr::Zipf has incompatible API)
        let am1 = a - 1.0;
        let b = 2.0_f64.powf(am1);
        let data: Vec<f64> = (0..size).map(|_| {
            loop {
                let u = 1.0 - self.next_f64();
                let v = self.next_f64();
                let x = (u.powf(-1.0 / am1)).floor();
                let t = (1.0 + 1.0 / x).powf(am1);
                if v * x * (t - 1.0) / (b - 1.0) <= t / b {
                    return x;
                }
            }
        }).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::int64())
    }

    // === Tier 3 Distributions: Specialized ===

    /// Generate samples from triangular distribution.
    pub fn triangular(&mut self, left: f64, mode: f64, right: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Triangular::new(left, right, mode).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from von Mises distribution.
    pub fn vonmises(&mut self, mu: f64, kappa: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        // rand_distr doesn't have vonmises, use Best-Fisher algorithm
        let data: Vec<f64> = (0..size).map(|_| {
            if kappa < 1e-6 {
                return std::f64::consts::PI * (2.0 * self.next_f64() - 1.0);
            }
            let tau = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
            let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * kappa);
            let r = (1.0 + rho * rho) / (2.0 * rho);
            loop {
                let u1 = self.next_f64();
                let z = (std::f64::consts::PI * u1).cos();
                let w = (1.0 + r * z) / (r + z);
                let c = kappa * (r - w);
                let u2 = self.next_f64();
                if c * (2.0 - c) - u2 > 0.0 || c.ln() - c + 1.0 - u2 >= 0.0 {
                    let sign = if self.next_f64() > 0.5 { 1.0 } else { -1.0 };
                    return mu + sign * w.acos();
                }
            }
        }).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from Pareto distribution.
    pub fn pareto(&mut self, a: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Pareto::new(1.0, a).unwrap();
        // NumPy's pareto returns (X - 1) where X ~ Pareto(1, a)
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self) - 1.0).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from Wald (inverse Gaussian) distribution.
    pub fn wald(&mut self, mean: f64, scale: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::InverseGaussian::new(mean, scale).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from Dirichlet distribution.
    pub fn dirichlet(&mut self, alpha: &[f64], size: usize) -> RumpyArray {
        let k = alpha.len();
        // Can't use rand_distr::Dirichlet (requires const generic array size)
        // Use Gamma sampling: sample Gamma(alpha_i, 1) and normalize
        let mut data = Vec::with_capacity(size * k);
        for _ in 0..size {
            let gamma_samples: Vec<f64> = alpha.iter()
                .map(|&a| self.next_standard_gamma(a))
                .collect();
            let sum: f64 = gamma_samples.iter().sum();
            data.extend(gamma_samples.iter().map(|&g| g / sum));
        }
        RumpyArray::from_vec_with_shape(data, vec![size, k], DType::float64())
    }

    /// Generate samples from standard Student's t distribution.
    pub fn standard_t(&mut self, df: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::StudentT::new(df).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from standard Cauchy distribution.
    pub fn standard_cauchy(&mut self, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Cauchy::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }

    /// Generate samples from standard gamma distribution (shape only, scale=1).
    pub fn standard_gamma_dist(&mut self, shape_param: f64, shape: Vec<usize>) -> RumpyArray {
        let size: usize = shape.iter().product();
        let dist = rand_distr::Gamma::new(shape_param, 1.0).unwrap();
        let data: Vec<f64> = (0..size).map(|_| dist.sample(self)).collect();
        RumpyArray::from_vec_with_shape(data, shape, DType::float64())
    }
}
