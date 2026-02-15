import numpy as np

EPS = 1e-5

def layernorm_forward(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    TODO:
    Implement LayerNorm over the last dimension.

    X: (N, D)
    gamma: (D,)
    beta: (D,)
    returns Y: (N, D)

    LayerNorm per row i:
      mu = mean(X[i,:])
      var = mean((X[i,:]-mu)^2)
      xhat = (X[i,:]-mu)/sqrt(var+eps)
      y = gamma * xhat + beta
    """
    # TODO
    raise NotImplementedError

def _assert_allclose(a, b, rtol=1e-6, atol=1e-6):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.max(np.abs(a-b))
        raise AssertionError(f"not close; max abs diff={diff}")

def main():
    X = np.array([[1.0, 2.0, 3.0],
                  [2.0, 4.0, 6.0]], dtype=np.float64)
    gamma = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    beta  = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    Y = layernorm_forward(X, gamma, beta, eps=1e-5)

    # Reference via direct numpy ops (this is what your implementation should match)
    mu = X.mean(axis=1, keepdims=True)
    var = ((X - mu) ** 2).mean(axis=1, keepdims=True)
    Y_ref = (X - mu) / np.sqrt(var + 1e-5)

    _assert_allclose(Y, Y_ref, rtol=1e-6, atol=1e-6)
    print("All tests passed!")

if __name__ == "__main__":
    main()
