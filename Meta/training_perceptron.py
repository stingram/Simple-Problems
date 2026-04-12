"""
Meta-style AI Coding Practice #8: Perceptron Training

Problem
-------
Implement binary perceptron training.

Write:

    def train_perceptron(X, y, epochs=10):
        ...

Input
-----
- X: NumPy array of shape (N, D)
- y: NumPy array of shape (N,), labels in {-1, +1}
- epochs: number of passes through the data

Model
-----
- weights w of shape (D,), initialized to zeros
- bias b initialized to 0.0

Prediction rule
---------------
For an example x_i:
    score = w · x_i + b
    pred = +1 if score >= 0 else -1

Update rule
-----------
If pred != y_i:
    w = w + y_i * x_i
    b = b + y_i

Return
------
Return:
    (w, b)

Requirements
------------
- Iterate through examples in given order, for each epoch
- Use NumPy
- Validate shapes
- Assume y contains only -1 and +1; raise if not

What this tests
---------------
- classical ML implementation
- simple training loop
- vector ops
- correctness / carefulness

Implement only train_perceptron().
Do not modify tests.
"""

import numpy as np


def train_perceptron(X, y, epochs=10):
    
    xN, xd = X.shape
    yN = y.shape[0]
    assert xN == yN
    assert np.where(y == -1)[0].shape[0] + np.where(y == 1)[0].shape[0] == yN
    
    w = np.zeros(xd)
    b = 0.0
    
    for _ in range(epochs):
        for i in range(xN):
            score =np.dot(w,X[i])+b
            y_pred = 1 if score >= 0 else -1
            if y_pred != y[i]:
                b += np.sum(y[i])
                w += y[i]*X[i]
    return w, b

# =========================
# Tests
# =========================

def test_shapes():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1, -1])

    w, b = train_perceptron(X, y, epochs=1)
    assert w.shape == (2,)
    assert isinstance(b, (int, float, np.floating))


def test_single_update():
    X = np.array([[1.0, 1.0]])
    y = np.array([-1])

    w, b = train_perceptron(X, y, epochs=1)

    # start with w=0, b=0 => score=0 => pred=+1, incorrect
    # update: w += -1 * [1,1], b += -1
    assert np.allclose(w, np.array([-1.0, -1.0]))
    assert np.isclose(b, -1.0)


def test_no_update_needed():
    X = np.array([[1.0, 0.0]])
    y = np.array([1])

    w, b = train_perceptron(X, y, epochs=0)
    assert np.allclose(w, np.zeros(1 if X.ndim == 1 else X.shape[1]))
    assert np.isclose(b, 0.0)


def test_linearly_separable():
    X = np.array([
        [ 2.0,  2.0],
        [ 1.0,  1.0],
        [-1.0, -1.0],
        [-2.0, -1.0],
    ])
    y = np.array([1, 1, -1, -1])

    w, b = train_perceptron(X, y, epochs=10)

    scores = X @ w + b
    preds = np.where(scores >= 0, 1, -1)
    assert np.array_equal(preds, y)


def test_invalid_labels():
    X = np.array([[1.0, 2.0]])
    y = np.array([0])

    try:
        train_perceptron(X, y)
        raise AssertionError("expected exception for invalid labels")
    except Exception:
        pass


def test_bad_shapes():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1])

    try:
        train_perceptron(X, y)
        raise AssertionError("expected exception for mismatched lengths")
    except Exception:
        pass


def run_all_tests():
    test_shapes()
    test_single_update()
    test_no_update_needed()
    test_linearly_separable()
    test_invalid_labels()
    test_bad_shapes()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()