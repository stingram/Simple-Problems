"""
Meta-style AI Coding Practice #9: Online Softmax (Numerically Stable)

Problem
-------
Implement softmax over a 1D vector using an ONLINE algorithm.

Write:

    def online_softmax(x):

Input
-----
- x: NumPy array of shape (N,)

Output
------
- softmax(x): NumPy array of shape (N,)

Constraint
----------
You must compute softmax in ONE PASS (streaming style),
without first computing max(x) separately.

Hint
----
Maintain:
    m = running max
    l = running normalization factor

Algorithm sketch:
    m_new = max(m, x_i)
    l = l * exp(m - m_new) + exp(x_i - m_new)

Then compute outputs in second pass OR store intermediates.

Requirements
------------
- Must be numerically stable
- Must match np.softmax within tolerance
- Cannot call np.max(x) first

What this tests
---------------
- numerical stability
- kernel-style thinking
- attention fundamentals

Implement only online_softmax().
"""

import numpy as np


def online_softmax(x):
    l = 0
    m = -np.inf
    for val in x:
        m_new = max(m, val)
        l = l*np.exp(m-m_new) + np.exp(val-m_new)
        m = m_new
    return np.exp(x-m)/l


# =========================
# Tests
# =========================

def test_basic():
    x = np.array([1.0, 2.0, 3.0])
    out = online_softmax(x)

    ref = np.exp(x) / np.sum(np.exp(x))
    assert np.allclose(out, ref, atol=1e-6)


def test_large_values():
    x = np.array([1000.0, 1001.0, 1002.0])
    out = online_softmax(x)

    ref = np.exp(x - np.max(x))
    ref = ref / np.sum(ref)

    assert np.allclose(out, ref, atol=1e-6)


def test_negative_values():
    x = np.array([-1000.0, -1001.0, -999.0])
    out = online_softmax(x)

    ref = np.exp(x - np.max(x))
    ref = ref / np.sum(ref)

    assert np.allclose(out, ref, atol=1e-6)


def test_single():
    x = np.array([42.0])
    out = online_softmax(x)
    assert np.allclose(out, np.array([1.0]))


def run_all_tests():
    test_basic()
    test_large_values()
    test_negative_values()
    test_single()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()