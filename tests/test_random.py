"""Tests for rumpy.random module.

Random module wraps Generator class with various distributions.
Tests verify shape, range, and statistical properties.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES
from helpers import assert_eq


# === Generator Creation ===


class TestGeneratorCreation:
    """Test creating Generator instances."""

    def test_default_rng(self):
        """Test that default_rng creates a Generator."""
        rng = rp.random.default_rng(42)
        assert rng is not None

    def test_generator_class(self):
        """Test creating Generator directly."""
        gen = rp.random.Generator(seed=42)
        val = gen.random()
        assert isinstance(val, float)

    def test_from_numpy_state(self):
        """Test creating Generator from numpy PCG64DXSM state."""
        # Get state from numpy
        bg = np.random.PCG64DXSM(42)
        state = bg.state['state']['state']
        inc = bg.state['state']['inc']

        # Create rumpy generator with same state
        rng = rp.random.Generator.from_numpy_state(state, inc)
        assert rng is not None


# === Reproducibility ===


class TestReproducibility:
    """Test that same seed gives same results."""

    def test_random_reproducible(self):
        rng1 = rp.random.default_rng(12345)
        rng2 = rp.random.default_rng(12345)

        arr1 = rng1.random(size=10)
        arr2 = rng2.random(size=10)

        np.testing.assert_array_equal(np.asarray(arr1), np.asarray(arr2))

    def test_integers_reproducible(self):
        rng1 = rp.random.default_rng(12345)
        rng2 = rp.random.default_rng(12345)

        arr1 = rng1.integers(0, 100, size=10)
        arr2 = rng2.integers(0, 100, size=10)

        np.testing.assert_array_equal(np.asarray(arr1), np.asarray(arr2))

    def test_normal_reproducible(self):
        rng1 = rp.random.default_rng(12345)
        rng2 = rp.random.default_rng(12345)

        arr1 = rng1.normal(0, 1, size=10)
        arr2 = rng2.normal(0, 1, size=10)

        np.testing.assert_array_equal(np.asarray(arr1), np.asarray(arr2))


# === Uniform Distribution (random) ===


class TestRandom:
    """Test rp.random.Generator.random() - uniform [0, 1)."""

    def test_scalar(self):
        """Test generating a single random float."""
        rng = rp.random.default_rng(42)
        val = rng.random()
        assert isinstance(val, float)
        assert 0.0 <= val < 1.0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.random(size=shape)
        assert arr.shape == shape

    def test_range(self):
        """Test all values in [0, 1)."""
        rng = rp.random.default_rng(42)
        arr = rng.random(size=1000)
        n = np.asarray(arr)
        assert np.all(n >= 0.0)
        assert np.all(n < 1.0)

    def test_exact_match_numpy(self):
        """Test exact match with numpy when using from_numpy_state."""
        seed = 42
        bg = np.random.PCG64DXSM(seed)
        state = bg.state['state']['state']
        inc = bg.state['state']['inc']

        rng_rp = rp.random.Generator.from_numpy_state(state, inc)
        rng_np = np.random.Generator(np.random.PCG64DXSM(seed))

        r = rng_rp.random(size=100)
        n = rng_np.random(size=100)
        assert_eq(r, n)


# === Integers ===


class TestIntegers:
    """Test rp.random.Generator.integers()."""

    def test_scalar_range(self):
        """Test generating single integer in [0, high)."""
        rng = rp.random.default_rng(42)
        val = rng.integers(10)
        assert isinstance(val, int)
        assert 0 <= val < 10

    def test_low_high_range(self):
        """Test generating integers in [low, high)."""
        rng = rp.random.default_rng(42)
        arr = rng.integers(5, 15, size=100)
        n = np.asarray(arr)
        assert np.all(n >= 5)
        assert np.all(n < 15)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.integers(0, 100, size=shape)
        assert arr.shape == shape

    def test_distribution_roughly_uniform(self):
        """Test distribution is roughly uniform."""
        rng = rp.random.default_rng(42)
        arr = rng.integers(0, 100, size=10000)
        n = np.asarray(arr)
        # Mean should be near 50 for uniform [0, 100)
        assert 45 < n.mean() < 55


# === Uniform (low, high) ===


class TestUniform:
    """Test rp.random.Generator.uniform()."""

    def test_scalar(self):
        """Test generating single uniform float."""
        rng = rp.random.default_rng(42)
        val = rng.uniform(2.0, 5.0)
        assert isinstance(val, float)
        assert 2.0 <= val < 5.0

    def test_range(self):
        """Test uniform in [low, high)."""
        rng = rp.random.default_rng(42)
        arr = rng.uniform(2.0, 5.0, size=1000)
        n = np.asarray(arr)
        assert np.all(n >= 2.0)
        assert np.all(n < 5.0)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.uniform(0.0, 1.0, size=shape)
        assert arr.shape == shape

    def test_exact_match_numpy(self):
        """Test exact match with numpy when using from_numpy_state."""
        seed = 456
        bg = np.random.PCG64DXSM(seed)
        state = bg.state['state']['state']
        inc = bg.state['state']['inc']

        rng_rp = rp.random.Generator.from_numpy_state(state, inc)
        rng_np = np.random.Generator(np.random.PCG64DXSM(seed))

        r = rng_rp.uniform(5.0, 10.0, size=100)
        n = rng_np.uniform(5.0, 10.0, size=100)
        assert_eq(r, n)


# === Normal Distribution ===


class TestNormal:
    """Test rp.random.Generator.normal()."""

    def test_scalar(self):
        """Test generating single normal value."""
        rng = rp.random.default_rng(42)
        val = rng.normal(0.0, 1.0)
        assert isinstance(val, float)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.normal(0.0, 1.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test normal has correct mean and std."""
        rng = rp.random.default_rng(42)
        arr = rng.normal(loc=10.0, scale=2.0, size=10000)
        n = np.asarray(arr)
        # Statistical test - should be approximately correct
        assert abs(n.mean() - 10.0) < 0.1
        assert abs(n.std() - 2.0) < 0.1

    @pytest.mark.parametrize("loc,scale", [(0.0, 1.0), (5.0, 2.0), (-3.0, 0.5)])
    def test_parameters(self, loc, scale):
        """Test different loc and scale parameters."""
        rng = rp.random.default_rng(42)
        arr = rng.normal(loc, scale, size=10000)
        n = np.asarray(arr)
        assert abs(n.mean() - loc) < 0.1
        assert abs(n.std() - scale) < 0.1


class TestStandardNormal:
    """Test rp.random.Generator.standard_normal()."""

    def test_scalar(self):
        """Test generating single standard normal value."""
        rng = rp.random.default_rng(42)
        val = rng.standard_normal()
        assert isinstance(val, float)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.standard_normal(size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test standard normal has mean 0 and std 1."""
        rng = rp.random.default_rng(42)
        arr = rng.standard_normal(size=10000)
        n = np.asarray(arr)
        assert abs(n.mean()) < 0.05
        assert abs(n.std() - 1.0) < 0.05


# === Exponential Distribution ===


class TestExponential:
    """Test rp.random.Generator.exponential()."""

    def test_scalar(self):
        """Test generating single exponential value."""
        rng = rp.random.default_rng(42)
        val = rng.exponential(1.0)
        assert isinstance(val, float)
        assert val > 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.exponential(1.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test exponential has correct mean."""
        rng = rp.random.default_rng(42)
        arr = rng.exponential(scale=2.0, size=10000)
        n = np.asarray(arr)
        # All values should be positive
        assert np.all(n > 0)
        # Mean should be approximately scale
        assert abs(n.mean() - 2.0) < 0.1

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 5.0])
    def test_scales(self, scale):
        """Test different scale parameters."""
        rng = rp.random.default_rng(42)
        arr = rng.exponential(scale, size=10000)
        n = np.asarray(arr)
        assert np.all(n > 0)
        assert abs(n.mean() - scale) < 0.2


class TestStandardExponential:
    """Test rp.random.Generator.standard_exponential()."""

    def test_scalar(self):
        """Test generating single standard exponential value."""
        rng = rp.random.default_rng(42)
        val = rng.standard_exponential()
        assert isinstance(val, float)
        assert val > 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.standard_exponential(size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test standard exponential has mean 1."""
        rng = rp.random.default_rng(42)
        arr = rng.standard_exponential(size=10000)
        n = np.asarray(arr)
        assert np.all(n > 0)
        assert abs(n.mean() - 1.0) < 0.05


# === Beta Distribution ===


class TestBeta:
    """Test rp.random.Generator.beta()."""

    def test_scalar(self):
        """Test generating single beta value."""
        rng = rp.random.default_rng(42)
        val = rng.beta(2.0, 5.0)
        assert isinstance(val, float)
        assert 0 < val < 1

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.beta(2.0, 5.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test beta has correct mean."""
        rng = rp.random.default_rng(42)
        a, b = 2.0, 5.0
        arr = rng.beta(a, b, size=10000)
        n = np.asarray(arr)

        # Beta mean = a / (a + b)
        expected_mean = a / (a + b)
        assert abs(n.mean() - expected_mean) < 0.01
        # Values should be in (0, 1)
        assert n.min() > 0
        assert n.max() < 1

    @pytest.mark.parametrize("a,b", [(1.0, 1.0), (2.0, 2.0), (5.0, 1.0), (0.5, 0.5)])
    def test_parameters(self, a, b):
        """Test different a and b parameters."""
        rng = rp.random.default_rng(42)
        arr = rng.beta(a, b, size=10000)
        n = np.asarray(arr)
        expected_mean = a / (a + b)
        assert abs(n.mean() - expected_mean) < 0.02
        assert n.min() > 0
        assert n.max() < 1


# === Gamma Distribution ===


class TestGamma:
    """Test rp.random.Generator.gamma()."""

    def test_scalar(self):
        """Test generating single gamma value."""
        rng = rp.random.default_rng(42)
        val = rng.gamma(2.0, 3.0)
        assert isinstance(val, float)
        assert val > 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.gamma(2.0, 3.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test gamma has correct mean."""
        rng = rp.random.default_rng(42)
        shape_param, scale = 2.0, 3.0
        arr = rng.gamma(shape_param, scale, size=10000)
        n = np.asarray(arr)

        # Gamma mean = shape * scale
        expected_mean = shape_param * scale
        assert abs(n.mean() - expected_mean) < 0.2
        # Values should be positive
        assert n.min() > 0

    @pytest.mark.parametrize("shape_param,scale", [(1.0, 1.0), (2.0, 1.0), (5.0, 2.0)])
    def test_parameters(self, shape_param, scale):
        """Test different shape and scale parameters."""
        rng = rp.random.default_rng(42)
        arr = rng.gamma(shape_param, scale, size=10000)
        n = np.asarray(arr)
        expected_mean = shape_param * scale
        assert abs(n.mean() - expected_mean) < 0.3
        assert n.min() > 0


# === Poisson Distribution ===


class TestPoisson:
    """Test rp.random.Generator.poisson()."""

    def test_scalar(self):
        """Test generating single poisson value."""
        rng = rp.random.default_rng(42)
        val = rng.poisson(5.0)
        assert isinstance(val, (int, float))
        assert val >= 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.poisson(5.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test poisson has correct mean."""
        rng = rp.random.default_rng(42)
        lam = 5.0
        arr = rng.poisson(lam, size=10000)
        n = np.asarray(arr)

        # Poisson mean = lambda
        assert abs(n.mean() - lam) < 0.15
        # Values should be non-negative integers
        assert n.min() >= 0

    @pytest.mark.parametrize("lam", [1.0, 5.0, 10.0, 20.0])
    def test_parameters(self, lam):
        """Test different lambda parameters."""
        rng = rp.random.default_rng(42)
        arr = rng.poisson(lam, size=10000)
        n = np.asarray(arr)
        assert abs(n.mean() - lam) < 0.3
        assert n.min() >= 0


# === Binomial Distribution ===


class TestBinomial:
    """Test rp.random.Generator.binomial()."""

    def test_scalar(self):
        """Test generating single binomial value."""
        rng = rp.random.default_rng(42)
        val = rng.binomial(10, 0.5)
        assert isinstance(val, (int, float))
        assert 0 <= val <= 10

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.binomial(10, 0.5, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test binomial has correct mean."""
        rng = rp.random.default_rng(42)
        n_trials, p = 10, 0.3
        arr = rng.binomial(n_trials, p, size=10000)
        result = np.asarray(arr)

        # Binomial mean = n * p
        expected_mean = n_trials * p
        assert abs(result.mean() - expected_mean) < 0.1
        # Values should be in [0, n]
        assert result.min() >= 0
        assert result.max() <= n_trials

    @pytest.mark.parametrize("n,p", [(10, 0.5), (20, 0.3), (5, 0.7)])
    def test_parameters(self, n, p):
        """Test different n and p parameters."""
        rng = rp.random.default_rng(42)
        arr = rng.binomial(n, p, size=10000)
        result = np.asarray(arr)
        expected_mean = n * p
        assert abs(result.mean() - expected_mean) < 0.2
        assert result.min() >= 0
        assert result.max() <= n


# === Chi-Square Distribution ===


class TestChisquare:
    """Test rp.random.Generator.chisquare()."""

    def test_scalar(self):
        """Test generating single chisquare value."""
        rng = rp.random.default_rng(42)
        val = rng.chisquare(5.0)
        assert isinstance(val, float)
        assert val > 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.chisquare(5.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test chisquare has correct mean."""
        rng = rp.random.default_rng(42)
        df = 5.0
        arr = rng.chisquare(df, size=10000)
        n = np.asarray(arr)

        # Chi-square mean = df
        assert abs(n.mean() - df) < 0.2
        # Values should be positive
        assert n.min() > 0

    @pytest.mark.parametrize("df", [1.0, 3.0, 5.0, 10.0])
    def test_parameters(self, df):
        """Test different degrees of freedom."""
        rng = rp.random.default_rng(42)
        arr = rng.chisquare(df, size=10000)
        n = np.asarray(arr)
        assert abs(n.mean() - df) < 0.3
        assert n.min() > 0


# === Multivariate Normal ===


class TestMultivariateNormal:
    """Test rp.random.Generator.multivariate_normal()."""

    def test_basic_2d(self):
        """Test 2D multivariate normal."""
        rng = rp.random.default_rng(42)
        mean = [1.0, 2.0]
        cov = [[1.0, 0.0], [0.0, 1.0]]
        arr = rng.multivariate_normal(mean, cov, size=1000)
        n = np.asarray(arr)

        assert n.shape == (1000, 2)
        # Check means are approximately correct
        assert abs(n[:, 0].mean() - 1.0) < 0.1
        assert abs(n[:, 1].mean() - 2.0) < 0.1

    def test_with_covariance(self):
        """Test multivariate normal with non-trivial covariance."""
        rng = rp.random.default_rng(42)
        mean = [1.0, 2.0]
        cov = [[1.0, 0.5], [0.5, 2.0]]
        arr = rng.multivariate_normal(mean, cov, size=10000)
        n = np.asarray(arr)

        assert n.shape == (10000, 2)
        # Check means
        assert abs(n[:, 0].mean() - 1.0) < 0.1
        assert abs(n[:, 1].mean() - 2.0) < 0.1

    def test_3d(self):
        """Test 3D multivariate normal."""
        rng = rp.random.default_rng(42)
        mean = [0.0, 1.0, 2.0]
        cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        arr = rng.multivariate_normal(mean, cov, size=100)
        n = np.asarray(arr)

        assert n.shape == (100, 3)


# === Permutation and Shuffle ===


class TestPermutation:
    """Test rp.random.Generator.permutation()."""

    def test_from_int(self):
        """Test permutation creates shuffled arange."""
        rng = rp.random.default_rng(42)
        p = rng.permutation(10)
        n = np.asarray(p)

        # Should be a permutation of 0-9
        assert n.shape == (10,)
        assert set(n.tolist()) == set(range(10))

    def test_from_array(self):
        """Test permutation shuffles an array."""
        rng = rp.random.default_rng(42)
        arr = rp.arange(10)
        p = rng.permutation(arr)
        n = np.asarray(p)

        # Should contain same elements
        assert n.shape == (10,)
        assert set(n.tolist()) == set(range(10))

    @pytest.mark.parametrize("n", [5, 10, 20, 100])
    def test_sizes(self, n):
        """Test different permutation sizes."""
        rng = rp.random.default_rng(42)
        p = rng.permutation(n)
        result = np.asarray(p)

        assert result.shape == (n,)
        assert set(result.tolist()) == set(range(n))


class TestShuffle:
    """Test rp.random.Generator.shuffle()."""

    def test_modifies_in_place(self):
        """Test shuffle modifies array in place."""
        rng = rp.random.default_rng(42)
        arr = rp.arange(10)
        original = np.asarray(arr).copy()

        rng.shuffle(arr)
        shuffled = np.asarray(arr)

        # Should contain same elements but be shuffled
        assert set(shuffled.tolist()) == set(original.tolist())
        # With high probability, order should differ
        assert not np.array_equal(shuffled, original)

    def test_2d_shuffle(self):
        """Test shuffle works on first axis of 2D array."""
        rng = rp.random.default_rng(42)
        arr = rp.arange(20).reshape(10, 2)
        original = np.asarray(arr).copy()

        rng.shuffle(arr)
        shuffled = np.asarray(arr)

        # Rows should be shuffled, but each row intact
        assert shuffled.shape == (10, 2)
        # Each row should still be a pair from original
        assert not np.array_equal(shuffled, original)


# === Choice ===


class TestChoice:
    """Test rp.random.Generator.choice()."""

    def test_from_int_scalar(self):
        """Test choice from range returns scalar."""
        rng = rp.random.default_rng(42)
        val = rng.choice(10)
        assert isinstance(val, float)
        assert 0 <= val < 10

    def test_from_int_array(self):
        """Test choice from range returns array."""
        rng = rp.random.default_rng(42)
        arr = rng.choice(10, size=(5,))
        n = np.asarray(arr)
        assert n.shape == (5,)
        assert np.all((n >= 0) & (n < 10))

    def test_from_array(self):
        """Test choice from array."""
        rng = rp.random.default_rng(42)
        values = rp.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = rng.choice(values, size=(10,))
        n = np.asarray(result)
        assert n.shape == (10,)
        # All values should be from original array
        assert all(v in [1.5, 2.5, 3.5, 4.5, 5.5] for v in n)

    def test_without_replacement(self):
        """Test choice without replacement."""
        rng = rp.random.default_rng(42)
        arr = rng.choice(10, size=(10,), replace=False)
        n = np.asarray(arr)
        # Should be a permutation (all unique)
        assert len(set(n.tolist())) == 10

    def test_with_replacement(self):
        """Test choice with replacement."""
        rng = rp.random.default_rng(42)
        arr = rng.choice(5, size=(100,), replace=True)
        n = np.asarray(arr)
        # With replacement, should see duplicates
        assert len(set(n.tolist())) < 100

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.choice(100, size=shape)
        assert arr.shape == shape

    def test_invalid_without_replacement(self):
        """Test error when sampling more than population without replacement."""
        rng = rp.random.default_rng(42)
        with pytest.raises(ValueError):
            rng.choice(5, size=(10,), replace=False)


# === Comparison with NumPy ===


class TestNumPyComparison:
    """Compare rumpy distributions against numpy (shape and statistical properties)."""

    def test_random_vs_numpy_shape(self):
        """Test random() produces same shape as numpy."""
        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.random(size=(3, 4, 5))
        n = rng_np.random(size=(3, 4, 5))

        assert np.asarray(r).shape == n.shape

    def test_integers_vs_numpy_range(self):
        """Test integers() matches numpy range."""
        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.integers(5, 15, size=(100,))
        n = rng_np.integers(5, 15, size=(100,))

        r_np = np.asarray(r)
        assert r_np.shape == n.shape
        # Same range constraints
        assert r_np.min() >= 5
        assert r_np.max() < 15
        assert n.min() >= 5
        assert n.max() < 15

    def test_normal_vs_numpy_statistics(self):
        """Test normal() produces same statistical properties as numpy."""
        loc, scale = 5.0, 2.0
        size = (10000,)

        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.normal(loc, scale, size=size)
        n = rng_np.normal(loc, scale, size=size)

        r_np = np.asarray(r)

        # Both should have approximately the same mean and std
        assert abs(r_np.mean() - loc) < 0.1
        assert abs(n.mean() - loc) < 0.1
        assert abs(r_np.std() - scale) < 0.1
        assert abs(n.std() - scale) < 0.1

    def test_exponential_vs_numpy_statistics(self):
        """Test exponential() matches numpy statistics."""
        scale = 3.0
        size = (10000,)

        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.exponential(scale, size=size)
        n = rng_np.exponential(scale, size=size)

        r_np = np.asarray(r)

        # Both should be positive with mean ~scale
        assert r_np.min() > 0
        assert n.min() > 0
        assert abs(r_np.mean() - scale) < 0.2
        assert abs(n.mean() - scale) < 0.2

    def test_beta_vs_numpy_shape(self):
        """Test beta matches numpy shape."""
        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.beta(2.0, 5.0, size=(10, 20))
        n = rng_np.beta(2.0, 5.0, size=(10, 20))

        assert np.asarray(r).shape == n.shape

    def test_gamma_vs_numpy_shape(self):
        """Test gamma matches numpy shape."""
        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.gamma(2.0, 3.0, size=(10, 20))
        n = rng_np.gamma(2.0, 3.0, size=(10, 20))

        assert np.asarray(r).shape == n.shape


# === Tier 1 Distributions: Common ===


class TestLognormal:
    """Test rp.random.Generator.lognormal()."""

    def test_scalar(self):
        """Test generating single lognormal value."""
        rng = rp.random.default_rng(42)
        val = rng.lognormal(0.0, 1.0)
        assert isinstance(val, float)
        assert val > 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.lognormal(0.0, 1.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test lognormal has correct mean."""
        rng = rp.random.default_rng(42)
        mean, sigma = 0.0, 0.5
        arr = rng.lognormal(mean, sigma, size=10000)
        n = np.asarray(arr)
        # Lognormal mean = exp(mean + sigma^2/2)
        expected_mean = np.exp(mean + sigma**2 / 2)
        assert abs(n.mean() - expected_mean) < 0.05
        assert n.min() > 0


class TestLaplace:
    """Test rp.random.Generator.laplace()."""

    def test_scalar(self):
        """Test generating single Laplace value."""
        rng = rp.random.default_rng(42)
        val = rng.laplace(0.0, 1.0)
        assert isinstance(val, float)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.laplace(0.0, 1.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test Laplace has correct mean."""
        rng = rp.random.default_rng(42)
        loc, scale = 2.0, 1.5
        arr = rng.laplace(loc, scale, size=10000)
        n = np.asarray(arr)
        assert abs(n.mean() - loc) < 0.1
        # Laplace variance = 2 * scale^2
        expected_var = 2 * scale**2
        assert abs(n.var() - expected_var) < 0.5


class TestLogistic:
    """Test rp.random.Generator.logistic()."""

    def test_scalar(self):
        """Test generating single logistic value."""
        rng = rp.random.default_rng(42)
        val = rng.logistic(0.0, 1.0)
        assert isinstance(val, float)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.logistic(0.0, 1.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test logistic has correct mean."""
        rng = rp.random.default_rng(42)
        loc, scale = 3.0, 2.0
        arr = rng.logistic(loc, scale, size=10000)
        n = np.asarray(arr)
        assert abs(n.mean() - loc) < 0.1


class TestRayleigh:
    """Test rp.random.Generator.rayleigh()."""

    def test_scalar(self):
        """Test generating single Rayleigh value."""
        rng = rp.random.default_rng(42)
        val = rng.rayleigh(1.0)
        assert isinstance(val, float)
        assert val >= 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.rayleigh(1.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test Rayleigh has correct mean."""
        rng = rp.random.default_rng(42)
        scale = 2.0
        arr = rng.rayleigh(scale, size=10000)
        n = np.asarray(arr)
        # Rayleigh mean = scale * sqrt(pi/2)
        expected_mean = scale * np.sqrt(np.pi / 2)
        assert abs(n.mean() - expected_mean) < 0.1
        assert n.min() >= 0


class TestWeibull:
    """Test rp.random.Generator.weibull()."""

    def test_scalar(self):
        """Test generating single Weibull value."""
        rng = rp.random.default_rng(42)
        val = rng.weibull(2.0)
        assert isinstance(val, float)
        assert val >= 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.weibull(2.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test Weibull values are positive."""
        rng = rp.random.default_rng(42)
        arr = rng.weibull(2.0, size=10000)
        n = np.asarray(arr)
        assert n.min() >= 0


# === Tier 2 Distributions: Discrete ===


class TestGeometric:
    """Test rp.random.Generator.geometric()."""

    def test_scalar(self):
        """Test generating single geometric value."""
        rng = rp.random.default_rng(42)
        val = rng.geometric(0.5)
        assert isinstance(val, (int, float))
        assert val >= 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.geometric(0.5, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test geometric has correct mean."""
        rng = rp.random.default_rng(42)
        p = 0.3
        arr = rng.geometric(p, size=10000)
        n = np.asarray(arr)
        # Geometric mean (failures before first success) = (1-p)/p
        expected_mean = (1 - p) / p
        assert abs(n.mean() - expected_mean) < 0.2
        assert n.min() >= 0


class TestNegativeBinomial:
    """Test rp.random.Generator.negative_binomial()."""

    def test_scalar(self):
        """Test generating single negative binomial value."""
        rng = rp.random.default_rng(42)
        val = rng.negative_binomial(5, 0.5)
        assert isinstance(val, (int, float))
        assert val >= 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.negative_binomial(5, 0.5, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test negative binomial has correct mean."""
        rng = rp.random.default_rng(42)
        n, p = 5, 0.4
        arr = rng.negative_binomial(n, p, size=10000)
        result = np.asarray(arr)
        # Negative binomial mean = n*(1-p)/p
        expected_mean = n * (1 - p) / p
        assert abs(result.mean() - expected_mean) < 0.5
        assert result.min() >= 0


class TestHypergeometric:
    """Test rp.random.Generator.hypergeometric()."""

    def test_scalar(self):
        """Test generating single hypergeometric value."""
        rng = rp.random.default_rng(42)
        val = rng.hypergeometric(10, 5, 7)
        assert isinstance(val, (int, float))
        assert 0 <= val <= 7

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.hypergeometric(10, 5, 7, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test hypergeometric has correct mean."""
        rng = rp.random.default_rng(42)
        ngood, nbad, nsample = 20, 10, 15
        arr = rng.hypergeometric(ngood, nbad, nsample, size=10000)
        n = np.asarray(arr)
        # Hypergeometric mean = nsample * ngood / (ngood + nbad)
        expected_mean = nsample * ngood / (ngood + nbad)
        assert abs(n.mean() - expected_mean) < 0.2
        assert n.min() >= 0
        assert n.max() <= nsample


class TestMultinomial:
    """Test rp.random.Generator.multinomial()."""

    def test_basic(self):
        """Test basic multinomial."""
        rng = rp.random.default_rng(42)
        result = rng.multinomial(10, [0.2, 0.3, 0.5])
        n = np.asarray(result)
        assert n.shape == (1, 3)
        # Each row should sum to n
        assert np.allclose(n.sum(axis=1), 10)

    def test_with_size(self):
        """Test multinomial with size parameter."""
        rng = rp.random.default_rng(42)
        result = rng.multinomial(10, [0.2, 0.3, 0.5], size=100)
        n = np.asarray(result)
        assert n.shape == (100, 3)
        assert np.allclose(n.sum(axis=1), 10)

    def test_statistics(self):
        """Test multinomial has correct means."""
        rng = rp.random.default_rng(42)
        n_trials = 100
        pvals = [0.2, 0.3, 0.5]
        result = rng.multinomial(n_trials, pvals, size=1000)
        arr = np.asarray(result)
        # Mean for each category = n * p
        for i, p in enumerate(pvals):
            expected = n_trials * p
            assert abs(arr[:, i].mean() - expected) < 2


class TestZipf:
    """Test rp.random.Generator.zipf()."""

    def test_scalar(self):
        """Test generating single Zipf value."""
        rng = rp.random.default_rng(42)
        val = rng.zipf(2.0)
        assert isinstance(val, (int, float))
        assert val >= 1

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.zipf(2.0, size=shape)
        assert arr.shape == shape

    def test_values_positive(self):
        """Test Zipf values are >= 1."""
        rng = rp.random.default_rng(42)
        arr = rng.zipf(2.0, size=1000)
        n = np.asarray(arr)
        assert n.min() >= 1


# === Tier 3 Distributions: Specialized ===


class TestTriangular:
    """Test rp.random.Generator.triangular()."""

    def test_scalar(self):
        """Test generating single triangular value."""
        rng = rp.random.default_rng(42)
        val = rng.triangular(0.0, 0.5, 1.0)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.triangular(0.0, 0.5, 1.0, size=shape)
        assert arr.shape == shape

    def test_range(self):
        """Test triangular values are in [left, right]."""
        rng = rp.random.default_rng(42)
        left, mode, right = 2.0, 5.0, 8.0
        arr = rng.triangular(left, mode, right, size=10000)
        n = np.asarray(arr)
        assert n.min() >= left
        assert n.max() <= right

    def test_statistics(self):
        """Test triangular has correct mean."""
        rng = rp.random.default_rng(42)
        left, mode, right = 0.0, 0.5, 1.0
        arr = rng.triangular(left, mode, right, size=10000)
        n = np.asarray(arr)
        # Triangular mean = (left + mode + right) / 3
        expected_mean = (left + mode + right) / 3
        assert abs(n.mean() - expected_mean) < 0.02


class TestVonmises:
    """Test rp.random.Generator.vonmises()."""

    def test_scalar(self):
        """Test generating single von Mises value."""
        rng = rp.random.default_rng(42)
        val = rng.vonmises(0.0, 1.0)
        assert isinstance(val, float)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.vonmises(0.0, 1.0, size=shape)
        assert arr.shape == shape

    def test_range(self):
        """Test von Mises values are in [-pi, pi]."""
        rng = rp.random.default_rng(42)
        arr = rng.vonmises(0.0, 1.0, size=1000)
        n = np.asarray(arr)
        assert n.min() >= -np.pi
        assert n.max() <= np.pi


class TestPareto:
    """Test rp.random.Generator.pareto()."""

    def test_scalar(self):
        """Test generating single Pareto value."""
        rng = rp.random.default_rng(42)
        val = rng.pareto(3.0)
        assert isinstance(val, float)
        assert val >= 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.pareto(3.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test Pareto values are non-negative."""
        rng = rp.random.default_rng(42)
        arr = rng.pareto(3.0, size=10000)
        n = np.asarray(arr)
        assert n.min() >= 0


class TestWald:
    """Test rp.random.Generator.wald()."""

    def test_scalar(self):
        """Test generating single Wald value."""
        rng = rp.random.default_rng(42)
        val = rng.wald(1.0, 1.0)
        assert isinstance(val, float)
        assert val > 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.wald(1.0, 1.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test Wald has correct mean."""
        rng = rp.random.default_rng(42)
        mean, scale = 2.0, 3.0
        arr = rng.wald(mean, scale, size=10000)
        n = np.asarray(arr)
        # Wald mean = mean parameter
        assert abs(n.mean() - mean) < 0.1
        assert n.min() > 0


class TestDirichlet:
    """Test rp.random.Generator.dirichlet()."""

    def test_basic(self):
        """Test basic Dirichlet."""
        rng = rp.random.default_rng(42)
        result = rng.dirichlet([1.0, 2.0, 3.0])
        n = np.asarray(result)
        assert n.shape == (1, 3)
        # Each row should sum to 1
        assert np.allclose(n.sum(axis=1), 1.0)
        # All values should be in [0, 1]
        assert n.min() >= 0
        assert n.max() <= 1

    def test_with_size(self):
        """Test Dirichlet with size parameter."""
        rng = rp.random.default_rng(42)
        result = rng.dirichlet([1.0, 2.0, 3.0], size=100)
        n = np.asarray(result)
        assert n.shape == (100, 3)
        assert np.allclose(n.sum(axis=1), 1.0)

    def test_statistics(self):
        """Test Dirichlet has correct means."""
        rng = rp.random.default_rng(42)
        alpha = [2.0, 3.0, 5.0]
        result = rng.dirichlet(alpha, size=10000)
        arr = np.asarray(result)
        # Mean for each category = alpha_i / sum(alpha)
        alpha_sum = sum(alpha)
        for i, a in enumerate(alpha):
            expected = a / alpha_sum
            assert abs(arr[:, i].mean() - expected) < 0.02


class TestStandardT:
    """Test rp.random.Generator.standard_t()."""

    def test_scalar(self):
        """Test generating single standard t value."""
        rng = rp.random.default_rng(42)
        val = rng.standard_t(5.0)
        assert isinstance(val, float)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.standard_t(5.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test standard t has mean near 0."""
        rng = rp.random.default_rng(42)
        df = 10.0
        arr = rng.standard_t(df, size=10000)
        n = np.asarray(arr)
        # t distribution mean = 0 for df > 1
        assert abs(n.mean()) < 0.1


class TestStandardCauchy:
    """Test rp.random.Generator.standard_cauchy()."""

    def test_scalar(self):
        """Test generating single standard Cauchy value."""
        rng = rp.random.default_rng(42)
        val = rng.standard_cauchy()
        assert isinstance(val, float)

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.standard_cauchy(size=shape)
        assert arr.shape == shape

    def test_median_near_zero(self):
        """Test standard Cauchy median is near 0."""
        rng = rp.random.default_rng(42)
        arr = rng.standard_cauchy(size=10000)
        n = np.asarray(arr)
        # Cauchy median = 0 (mean is undefined)
        assert abs(np.median(n)) < 0.1


class TestStandardGamma:
    """Test rp.random.Generator.standard_gamma()."""

    def test_scalar(self):
        """Test generating single standard gamma value."""
        rng = rp.random.default_rng(42)
        val = rng.standard_gamma(2.0)
        assert isinstance(val, float)
        assert val > 0

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various output shapes."""
        rng = rp.random.default_rng(42)
        arr = rng.standard_gamma(2.0, size=shape)
        assert arr.shape == shape

    def test_statistics(self):
        """Test standard gamma has correct mean."""
        rng = rp.random.default_rng(42)
        shape_param = 3.0
        arr = rng.standard_gamma(shape_param, size=10000)
        n = np.asarray(arr)
        # Standard gamma mean = shape
        assert abs(n.mean() - shape_param) < 0.15
        assert n.min() > 0


# === NumPy Comparison for New Distributions ===


class TestNewDistributionsNumPyComparison:
    """Compare new distributions against numpy (shape and statistical properties)."""

    def test_lognormal_vs_numpy(self):
        """Test lognormal matches numpy statistics."""
        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.lognormal(0.0, 1.0, size=(1000,))
        n = rng_np.lognormal(0.0, 1.0, size=(1000,))

        r_np = np.asarray(r)
        # Both should have similar statistics (not exact match due to different algorithms)
        assert r_np.shape == n.shape
        assert r_np.min() > 0
        assert n.min() > 0

    def test_laplace_vs_numpy(self):
        """Test Laplace matches numpy shape."""
        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.laplace(0.0, 1.0, size=(10, 20))
        n = rng_np.laplace(0.0, 1.0, size=(10, 20))

        assert np.asarray(r).shape == n.shape

    def test_triangular_vs_numpy(self):
        """Test triangular matches numpy shape and range."""
        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.triangular(0.0, 0.5, 1.0, size=(1000,))
        n = rng_np.triangular(0.0, 0.5, 1.0, size=(1000,))

        r_np = np.asarray(r)
        assert r_np.shape == n.shape
        assert r_np.min() >= 0.0
        assert r_np.max() <= 1.0
        assert n.min() >= 0.0
        assert n.max() <= 1.0


# === Edge Cases ===


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_choice(self):
        """Test error on empty population."""
        rng = rp.random.default_rng(42)
        with pytest.raises(ValueError):
            rng.choice(0)

    def test_zero_size(self):
        """Test generating zero-size arrays."""
        rng = rp.random.default_rng(42)
        arr = rng.random(size=(0,))
        assert arr.shape == (0,)
        assert arr.size == 0

    def test_scalar_no_size(self):
        """Test that no size parameter returns scalar."""
        rng = rp.random.default_rng(42)
        val = rng.random()
        assert isinstance(val, float)
        assert not hasattr(val, 'shape')
