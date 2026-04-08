"""
Meta-style AI Coding Practice #4: 1D Convolution

Problem
-------
Implement a 1D convolution layer (no padding, stride=1).

You are given:
- x: input tensor of shape (N, C_in, L)
- w: weight tensor of shape (C_out, C_in, K)

Return:
- output tensor of shape (N, C_out, L_out)

Where:
    L_out = L - K + 1

Definition
----------
For each batch n, output channel c_out, and position t:

    out[n, c_out, t] =
        sum over c_in and k:
            x[n, c_in, t + k] * w[c_out, c_in, k]

Requirements
------------
- No padding
- Stride = 1
- Use nested loops (no torch.conv1d)
- Must handle arbitrary N, C_in, C_out
- Keep implementation clean and readable

What this tests
---------------
- tensor indexing
- loop structure
- shape reasoning
- correctness discipline

Implement only conv1d_naive().
"""

import math
def _dot_product_sum(x,w) -> float:
    acc = 0
    for xv,wv in zip(x,w):
        acc += xv*wv
    return acc

def conv1d_naive(x, w):
    N, Cin, L = len(x), len(x[0]), len(x[0][0])
    Cout, Cinw, K = len(w), len(w[0]), len(w[0][0])
    assert Cin == Cinw
    P = 0
    S = 1
    outL = math.floor((L - K - 2*P)/S) + 1
    out = [[[0 for _ in range(outL)] for _ in range(Cout)] for _ in range(N)]
    print(f'{N=},{Cin=},{L=},{Cout=},{Cin=},{K=},{outL=}')
    
    for batch in range(N):
        for cout in range(Cout):
            for out_i in range(outL):
                acc = 0
                for cin in range(Cin):
                    acc += _dot_product_sum(x[batch][cin][out_i:out_i+K],w[cout][cin][:])
                print(f'{acc=}')
                out[batch][cout][out_i] = acc
    print(f'{out=}')
    return out

# =========================
# Tests
# =========================

def test_simple():
    x = [[[1,2,3,4]]]  # (1,1,4)
    w = [[[1,1]]]      # (1,1,2)

    out = conv1d_naive(x, w)
    assert out == [[[3,5,7]]]


def test_multi_channel():
    x = [[[1,2,3],[4,5,6]]]  # (1,2,3)
    w = [[[1,0],[0,1]]]      # (1,2,2)

    out = conv1d_naive(x, w)
    # expected:
    # t=0: 1*1 + 2*0 + 4*0 + 5*1 = 6
    # t=1: 2*1 + 3*0 + 5*0 + 6*1 = 8
    assert out == [[[6,8]]]


def test_multi_out_channel():
    x = [[[1,2,3,4]]]
    w = [
        [[1,1]],
        [[2,0]],
    ]

    out = conv1d_naive(x, w)
    assert out == [[[3,5,7],[2,4,6]]]


def test_batch():
    x = [
        [[1,2,3]],
        [[4,5,6]],
    ]
    w = [[[1,1]]]

    out = conv1d_naive(x, w)
    assert out == [[[3,5]], [[9,11]]]


def run_all_tests():
    test_simple()
    test_multi_channel()
    test_multi_out_channel()
    test_batch()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()