"""
Meta-style AI Coding Practice #7: Symmetric Per-Tensor Post-Training Quantization

Problem
-------
Implement symmetric int8 quantization for a floating-point weight tensor.

Write:

    def quantize_int8_symmetric(x):
        ...

Input
-----
x: NumPy array of arbitrary shape, dtype float

Output
------
Return a tuple:

    (q, scale)

where
- q is a NumPy array of same shape as x, dtype np.int8
- scale is a positive float

Quantization rule
-----------------
Use symmetric per-tensor quantization with int8 range [-127, 127].

Let:

    max_abs = max(abs(x))

If max_abs == 0:
    return zeros_like(x, dtype=int8), 1.0

Else:
    scale = max_abs / 127.0

Quantize each element as:

    q = round(x / scale)

Then clip to [-127, 127], cast to int8.

Notes
-----
- Use -127..127, not -128..127
- round should be NumPy round
- per-tensor means one shared scale for all elements

What this tests
---------------
- quantization basics
- array ops
- edge cases
- careful reading

Implement only quantize_int8_symmetric().
Do not modify tests.
"""

import numpy as np


def quantize_int8_symmetric(x):
    maxx = np.max(np.abs(x))
    if maxx == 0:
        return np.array(x,dtype=np.int8), 1.0
    scale = maxx / 127.0
    return np.array(np.clip(np.round(x/scale),-127,127),dtype=np.int8), scale


# =========================
# Tests
# =========================

def test_basic():
    x = np.array([-1.0, 0.0, 1.0], dtype=float)
    q, scale = quantize_int8_symmetric(x)

    assert q.dtype == np.int8
    assert q.shape == x.shape
    assert np.isclose(scale, 1.0 / 127.0)
    assert np.array_equal(q, np.array([-127, 0, 127], dtype=np.int8))


def test_zero_tensor():
    x = np.zeros((2, 3), dtype=float)
    q, scale = quantize_int8_symmetric(x)

    assert q.dtype == np.int8
    assert np.array_equal(q, np.zeros_like(x, dtype=np.int8))
    assert scale == 1.0


def test_clip_behavior():
    x = np.array([-1000.0, 0.0, 1000.0], dtype=float)
    q, scale = quantize_int8_symmetric(x)

    assert np.array_equal(q, np.array([-127, 0, 127], dtype=np.int8))


def test_mixed_values():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    q, scale = quantize_int8_symmetric(x)

    expected_scale = 2.0 / 127.0
    expected_q = np.array([-127, -64, 0, 64, 127], dtype=np.int8)

    assert np.isclose(scale, expected_scale)
    assert np.array_equal(q, expected_q)


def test_multidim():
    x = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=float)
    q, scale = quantize_int8_symmetric(x)

    assert q.shape == x.shape
    assert q.dtype == np.int8
    assert np.isclose(scale, 1.0 / 127.0)


def run_all_tests():
    test_basic()
    test_zero_tensor()
    test_clip_behavior()
    test_mixed_values()
    test_multidim()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()