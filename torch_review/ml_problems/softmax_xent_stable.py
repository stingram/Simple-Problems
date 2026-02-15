import numpy as np

EPS = 1e-12

def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """
    TODO:
    Compute numerically-stable softmax over the last dimension.

    logits: shape (..., C)
    returns: same shape (..., C), probabilities sum to 1 along last dim.

    Requirements:
      - subtract max per row (along last dim) for stability
      - use np.exp
    """
    maxs = np.max(logits,axis=-1,keepdims=True)
    numerator = np.exp(logits-maxs)
    denominator = np.sum(numerator,axis=-1,keepdims=True)
    return numerator / denominator

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    TODO:
    Compute average cross-entropy loss for integer labels.

    logits: shape (N, C)
    labels: shape (N,) int in [0, C)
    returns: scalar float = (1/N) sum_i -log( softmax(logits)[i, labels[i]] )

    Requirements:
      - use stable softmax
      - clamp probabilities with EPS to avoid log(0)
    """
    pred = softmax_stable(logits)
    pred = np.clip(pred, EPS, 1.0)
    losses = -np.log(pred[np.arange(pred.shape[-2]),labels])
    return np.mean(losses)
    


def _assert_close(a, b, tol=1e-6):
    if abs(a - b) > tol:
        raise AssertionError(f"{a} != {b} within tol={tol}")

def main():
    # Test 1: simple known case
    # Sample 0: [0,0,0] => p=[1/3,1/3,1/3], y=0 => -log(1/3)=1.0986122887
    # Sample 1: [2,0,-2] => p0 ~ 0.866813... => -log(p0)=0.1429316285
    # avg ~ 0.6207719586
    logits = np.array([[0.0, 0.0, 0.0],
                       [2.0, 0.0, -2.0]], dtype=np.float64)
    labels = np.array([0, 0], dtype=np.int64)
    loss = cross_entropy_loss(logits, labels)
    _assert_close(loss, 0.6207719586, 1e-6)

    # Test 2: stability large logits
    p = softmax_stable(np.array([1000.0, 999.0], dtype=np.float64))
    _assert_close(float(p[0]), 0.7310585786, 1e-6)
    _assert_close(float(p[1]), 0.2689414214, 1e-6)
    _assert_close(float(p.sum()), 1.0, 1e-9)

    print("All tests passed!")

if __name__ == "__main__":
    main()