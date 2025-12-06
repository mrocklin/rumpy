"""Time series analysis benchmarks.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def moving_average(xp, size):
    """Simple moving average with window."""
    data = np.cumsum(np.random.randn(size))  # random walk
    x = xp.array(data)
    window = 10

    def fn():
        # Cumsum-based moving average
        cumsum = xp.cumsum(x)
        # Requires advanced indexing: cumsum[window:] - cumsum[:-window]
        return (cumsum[window:] - cumsum[:-window]) / window
    return fn


def exponential_smoothing(xp, size):
    """Exponential moving average (approximate via vectorized ops)."""
    data = np.random.randn(size)
    x = xp.array(data)
    alpha = 0.1

    def fn():
        # Compute weighted sum with exponential decay
        # This is an approximation - true EMA needs scan
        weights = alpha * (1 - alpha) ** xp.arange(size)
        # Use outer product and sum
        # Simplified: just compute weighted recent values
        return (x * weights).sum()
    return fn


def autocorrelation(xp, size):
    """Autocorrelation via FFT."""
    data = np.random.randn(size)
    x = xp.array(data)

    def fn():
        # Autocorrelation via FFT: ifft(|fft(x)|^2)
        f = xp.fft.fft(x)
        power = xp.real(f * xp.conj(f))
        return xp.fft.ifft(power)
    return fn


def detrend_linear(xp, size):
    """Remove linear trend from signal."""
    data = np.linspace(0, 10, size) + np.random.randn(size)
    x = xp.array(data)
    t = xp.arange(size, dtype='float64')

    def fn():
        # Linear detrend: x - (a*t + b)
        # Solve for a, b using least squares
        n = size
        sum_t = t.sum()
        sum_x = x.sum()
        sum_tt = (t * t).sum()
        sum_tx = (t * x).sum()
        a = (n * sum_tx - sum_t * sum_x) / (n * sum_tt - sum_t * sum_t)
        b = (sum_x - a * sum_t) / n
        return x - (a * t + b)
    return fn


def rolling_std(xp, size):
    """Rolling standard deviation."""
    data = np.random.randn(size)
    x = xp.array(data)
    window = 20

    def fn():
        # Compute using cumsum for mean and variance
        cumsum = xp.cumsum(x)
        cumsum_sq = xp.cumsum(x * x)
        # Rolling sums
        roll_sum = cumsum[window:] - cumsum[:-window]
        roll_sum_sq = cumsum_sq[window:] - cumsum_sq[:-window]
        # Variance = E[x^2] - E[x]^2
        mean = roll_sum / window
        mean_sq = roll_sum_sq / window
        var = mean_sq - mean * mean
        return xp.sqrt(xp.maximum(var, 0))  # clamp negative due to numerical
    return fn


def returns_from_prices(xp, size):
    """Compute log returns from price series."""
    prices = np.cumsum(np.abs(np.random.randn(size))) + 100
    x = xp.array(prices)

    def fn():
        return xp.log(x[1:] / x[:-1])
    return fn


def zscore_rolling(xp, size):
    """Rolling z-score normalization."""
    data = np.random.randn(size)
    x = xp.array(data)
    window = 50

    def fn():
        cumsum = xp.cumsum(x)
        cumsum_sq = xp.cumsum(x * x)
        # Rolling stats
        roll_sum = cumsum[window:] - cumsum[:-window]
        roll_sum_sq = cumsum_sq[window:] - cumsum_sq[:-window]
        mean = roll_sum / window
        var = roll_sum_sq / window - mean * mean
        std = xp.sqrt(xp.maximum(var, 1e-10))
        # Z-score for values from window onwards
        return (x[window:] - mean) / std
    return fn


def lag_features(xp, size):
    """Create lagged features for time series."""
    data = np.random.randn(size)
    x = xp.array(data)
    max_lag = 10

    def fn():
        # Compute lagged differences (simplified lag features)
        # x[t] - x[t-1], x[t] - x[t-2], etc.
        base = x[max_lag:]
        lag1 = x[max_lag-1:-1]
        lag2 = x[max_lag-2:-2]
        lag5 = x[max_lag-5:-5]
        lag10 = x[:-max_lag]
        return (base - lag1) + (base - lag2) + (base - lag5) + (base - lag10)
    return fn


def seasonal_decompose_simple(xp, size):
    """Simple seasonal decomposition (trend + residual)."""
    # Create data with trend + seasonality + noise
    t = np.arange(size)
    trend = 0.01 * t
    season = np.sin(2 * np.pi * t / 12)  # monthly seasonality
    noise = np.random.randn(size) * 0.1
    data = trend + season + noise
    x = xp.array(data)
    period = 12

    def fn():
        # Estimate trend with moving average
        cumsum = xp.cumsum(x)
        # Pad-aware moving average
        trend_est = (cumsum[period:] - cumsum[:-period]) / period
        # Residual (can't perfectly align without more ops)
        return trend_est
    return fn


def crosscorr(xp, size):
    """Cross-correlation between two signals via FFT."""
    a = xp.array(np.random.randn(size))
    b = xp.array(np.random.randn(size))

    def fn():
        fa = xp.fft.fft(a)
        fb = xp.fft.fft(b)
        return xp.fft.ifft(fa * xp.conj(fb))
    return fn


def ewma_weights(xp, size):
    """Compute EWMA weights."""
    span = 10
    alpha = 2.0 / (span + 1)

    def fn():
        idx = xp.arange(size, dtype='float64')
        weights = (1 - alpha) ** idx
        return weights / weights.sum()
    return fn


BENCHMARKS = [
    ("moving average", moving_average),
    ("exponential smoothing", exponential_smoothing),
    ("autocorrelation (FFT)", autocorrelation),
    ("detrend linear", detrend_linear),
    ("rolling std", rolling_std),
    ("log returns", returns_from_prices),
    ("rolling z-score", zscore_rolling),
    ("lag features", lag_features),
    ("seasonal decompose", seasonal_decompose_simple),
    ("cross-correlation", crosscorr),
    ("EWMA weights", ewma_weights),
]
