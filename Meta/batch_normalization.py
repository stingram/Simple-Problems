"""
Meta-style AI Coding Practice #5: Batch Normalization (Training Mode Only)

Problem
-------
Implement batch normalization for a 2D input in training mode.

You are given:
- x: input array of shape (N, D)
- gamma: scale parameter of shape (D,)
- beta: shift parameter of shape (D,)
- eps: small constant for numerical stability

Return:
- out: normalized output of shape (N, D)
- batch_mean: mean used for normalization, shape (D,)
- batch_var: variance used for normalization, shape (D,)

Definition
----------
For each feature dimension d:

    mean[d] = average over batch dimension
    var[d]  = average of squared deviations over batch dimension  (use population variance)

    x_hat[:, d] = (x[:, d] - mean[d]) / sqrt(var[d] + eps)
    out[:, d] = gamma[d] * x_hat[:, d] + beta[d]

Requirements
------------
- Use NumPy
- Do not call a batchnorm implementation from a library
- Use population variance (divide by N, not N-1)
- Return batch_mean and batch_var exactly as used
- Input is always 2D

What this tests
---------------
- vectorized NumPy reasoning
- numerics
- shape/broadcasting
- correctness of normalization

Implement only batchnorm_forward().
Do not modify the tests.
"""

import numpy as np


def batchnorm_forward(x, gamma, beta, eps=1e-5):
    xmean = np.mean(x,axis=0)
    xvar = np.var(x,axis=0)
    
    x_hat = (x - xmean) / np.sqrt(xvar + eps)
    out = x_hat*gamma + beta
    return (out, xmean, xvar)
    


# =========================
# Tests
# =========================

def test_basic_shape():
    x = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    gamma = np.array([1.0, 1.0])
    beta = np.array([0.0, 0.0])

    out, mean, var = batchnorm_forward(x, gamma, beta)

    assert out.shape == (2, 2)
    assert mean.shape == (2,)
    assert var.shape == (2,)


def test_mean_and_var():
    x = np.array([[1.0, 2.0],
                  [3.0, 6.0]])
    gamma = np.array([1.0, 1.0])
    beta = np.array([0.0, 0.0])

    out, mean, var = batchnorm_forward(x, gamma, beta, eps=0.0)

    expected_mean = np.array([2.0, 4.0])
    expected_var = np.array([1.0, 4.0])

    assert np.allclose(mean, expected_mean)
    assert np.allclose(var, expected_var)

    expected_out = np.array([[-1.0, -1.0],
                             [ 1.0,  1.0]])
    assert np.allclose(out, expected_out)


def test_gamma_beta():
    x = np.array([[1.0, 2.0],
                  [3.0, 6.0]])
    gamma = np.array([2.0, 3.0])
    beta = np.array([10.0, 20.0])

    out, mean, var = batchnorm_forward(x, gamma, beta, eps=0.0)

    expected = np.array([[ 8.0, 17.0],
                         [12.0, 23.0]])
    assert np.allclose(out, expected)


def test_zero_variance():
    x = np.array([[5.0, 1.0],
                  [5.0, 1.0],
                  [5.0, 1.0]])
    gamma = np.array([1.0, 2.0])
    beta = np.array([0.0, 3.0])

    out, mean, var = batchnorm_forward(x, gamma, beta)

    expected_mean = np.array([5.0, 1.0])
    expected_var = np.array([0.0, 0.0])

    assert np.allclose(mean, expected_mean)
    assert np.allclose(var, expected_var)

    # normalized term should be zero because x == mean
    expected_out = np.array([[0.0, 3.0],
                             [0.0, 3.0],
                             [0.0, 3.0]])
    assert np.allclose(out, expected_out)


def test_input_validation():
    x = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    gamma = np.array([1.0])
    beta = np.array([0.0, 0.0])

    try:
        batchnorm_forward(x, gamma, beta)
        raise AssertionError("expected exception for bad gamma shape")
    except Exception:
        pass


def run_all_tests():
    test_basic_shape()
    test_mean_and_var()
    test_gamma_beta()
    test_zero_variance()
    test_input_validation()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()