"""Machine learning primitive benchmarks.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def linear_layer(xp, size):
    """Linear layer: y = x @ W + b"""
    batch = int(np.sqrt(size))
    in_features = 64
    out_features = 32
    x = xp.array(np.random.randn(batch, in_features))
    W = xp.array(np.random.randn(in_features, out_features))
    b = xp.array(np.random.randn(out_features))

    def fn():
        return x @ W + b
    return fn


def batch_norm_inference(xp, size):
    """Batch normalization inference: (x - mean) / sqrt(var + eps) * gamma + beta"""
    batch = int(np.sqrt(size))
    features = 64
    x = xp.array(np.random.randn(batch, features))
    running_mean = xp.array(np.random.randn(features))
    running_var = xp.array(np.abs(np.random.randn(features)) + 0.1)
    gamma = xp.array(np.random.randn(features))
    beta = xp.array(np.random.randn(features))
    eps = 1e-5

    def fn():
        return (x - running_mean) / xp.sqrt(running_var + eps) * gamma + beta
    return fn


def attention_scores(xp, size):
    """Attention score computation: softmax(Q @ K.T / sqrt(d))"""
    seq_len = int(np.sqrt(size))
    d_k = 64
    Q = xp.array(np.random.randn(seq_len, d_k))
    K = xp.array(np.random.randn(seq_len, d_k))
    scale = np.sqrt(d_k)

    def fn():
        scores = Q @ K.T / scale
        # Softmax along last axis
        scores_max = scores.max(axis=1, keepdims=True)
        exp_scores = xp.exp(scores - scores_max)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return fn


def cross_entropy_loss(xp, size):
    """Cross entropy loss: -sum(y_true * log(y_pred))"""
    batch = size
    num_classes = 10
    # Logits
    logits = xp.array(np.random.randn(batch, num_classes))
    # One-hot labels
    labels_idx = np.random.randint(0, num_classes, batch)
    labels = np.zeros((batch, num_classes))
    labels[np.arange(batch), labels_idx] = 1.0
    y_true = xp.array(labels)

    def fn():
        # Softmax
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = xp.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Cross entropy
        log_probs = xp.log(probs + 1e-10)
        return -(y_true * log_probs).sum() / batch
    return fn


def mse_loss(xp, size):
    """Mean squared error loss."""
    y_pred = xp.array(np.random.randn(size))
    y_true = xp.array(np.random.randn(size))

    def fn():
        diff = y_pred - y_true
        return (diff * diff).mean()
    return fn


def gradient_descent_step(xp, size):
    """One gradient descent step: w = w - lr * grad"""
    w = xp.array(np.random.randn(size))
    grad = xp.array(np.random.randn(size))
    lr = 0.01

    def fn():
        return w - lr * grad
    return fn


def adam_update(xp, size):
    """Adam optimizer update step."""
    w = xp.array(np.random.randn(size))
    grad = xp.array(np.random.randn(size))
    m = xp.array(np.random.randn(size) * 0.1)
    v = xp.array(np.abs(np.random.randn(size)) * 0.01)
    lr = 0.001
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 100  # step number

    def fn():
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m_new / (1 - beta1**t)
        v_hat = v_new / (1 - beta2**t)
        return w - lr * m_hat / (xp.sqrt(v_hat) + eps)
    return fn


def dropout_mask(xp, size):
    """Create dropout mask (requires random)."""
    rng = xp.random.default_rng(42)
    x = xp.array(np.random.randn(size))
    p = 0.5

    def fn():
        mask = rng.random(size) > p
        return x * mask / (1 - p)
    return fn


def embedding_lookup(xp, size):
    """Embedding lookup (integer indexing)."""
    vocab_size = 10000
    embed_dim = 128
    seq_len = size
    embeddings = xp.array(np.random.randn(vocab_size, embed_dim))
    indices = np.random.randint(0, vocab_size, seq_len)
    idx = xp.array(indices)

    def fn():
        # Integer array indexing
        return embeddings[idx]
    return fn


def cosine_similarity(xp, size):
    """Cosine similarity between two vectors."""
    a = xp.array(np.random.randn(size))
    b = xp.array(np.random.randn(size))

    def fn():
        dot = (a * b).sum()
        norm_a = xp.sqrt((a * a).sum())
        norm_b = xp.sqrt((b * b).sum())
        return dot / (norm_a * norm_b)
    return fn


def l2_regularization(xp, size):
    """L2 regularization term: lambda * sum(w^2)"""
    w = xp.array(np.random.randn(size))
    lambda_reg = 0.01

    def fn():
        return lambda_reg * (w * w).sum()
    return fn


def mlp_forward(xp, size):
    """MLP forward pass: 2 linear layers with ReLU."""
    batch = int(np.sqrt(size))
    h1, h2, h3 = 64, 32, 10
    x = xp.array(np.random.randn(batch, h1))
    W1 = xp.array(np.random.randn(h1, h2))
    b1 = xp.array(np.random.randn(h2))
    W2 = xp.array(np.random.randn(h2, h3))
    b2 = xp.array(np.random.randn(h3))
    zero = xp.zeros(1)

    def fn():
        h = xp.maximum(x @ W1 + b1, zero)  # ReLU
        return h @ W2 + b2
    return fn


BENCHMARKS = [
    ("linear layer", linear_layer),
    ("batch norm inference", batch_norm_inference),
    ("attention scores", attention_scores),
    ("cross entropy loss", cross_entropy_loss),
    ("MSE loss", mse_loss),
    ("gradient descent step", gradient_descent_step),
    ("Adam update", adam_update),
    ("dropout mask", dropout_mask),
    ("embedding lookup", embedding_lookup),  # requires integer array indexing
    ("cosine similarity", cosine_similarity),
    ("L2 regularization", l2_regularization),
    ("MLP forward (2 layers)", mlp_forward),
]
