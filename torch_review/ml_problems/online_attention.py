import math
import numpy as np

# ============================================================
# TODO: Implement this
# ============================================================

def online_attention(Q, K, V, block_k: int = 128, causal: bool = False):
    """
    Blockwise attention using online (streaming) numerically-stable softmax.

    Args:
      Q: (B, Tq, D) float32/float64
      K: (B, Tk, D)
      V: (B, Tk, Dv)
      block_k: keys/values block size along Tk
      causal: if True, apply causal mask (for each query position t, keys > t are masked)

    Returns:
      O: (B, Tq, Dv)
    """
    # ---- sanity ----
    assert Q.ndim == 3 and K.ndim == 3 and V.ndim == 3
    B, Tq, D = Q.shape
    B2, Tk, D2 = K.shape
    B3, Tk2, Dv = V.shape
    assert B == B2 == B3
    assert D == D2
    assert Tk == Tk2

    scale = 1.0 / math.sqrt(D)

    # Running stats per (B, Tq):
    # m: running max of scores
    # l: running sumexp in max-shifted domain
    # o: running output vector
    m = np.full((B, Tq), -np.inf, dtype=Q.dtype)
    l = np.zeros((B, Tq), dtype=Q.dtype)
    o = np.zeros((B, Tq, Dv), dtype=Q.dtype)

    # Iterate blocks of K,V
    for start in range(0, Tk, block_k):
        end = min(start + block_k, Tk)
        K_blk = K[:, start:end, :]          # (B, Bk, D)
        V_blk = V[:, start:end, :]          # (B, Bk, Dv)

        # Scores for this block: S_blk[b, tq, bk] = (Q[b,tq] dot K_blk[b,bk]) * scale
        # TODO: compute S_blk without huge temporaries beyond this block
        # shape (B, Tq, Bk)
        S_blk = None  # TODO

        # Apply causal masking (mask keys where (start + bk) > tq)
        if causal:
            # Build mask of shape (Tq, Bk): True means masked
            # For query index t, valid keys satisfy key_index <= t
            key_idx = np.arange(start, end)[None, :]          # (1, Bk)
            q_idx = np.arange(Tq)[:, None]                    # (Tq, 1)
            mask = key_idx > q_idx                            # (Tq, Bk)
            # Broadcast to (B, Tq, Bk)
            S_blk = np.where(mask[None, :, :], -np.inf, S_blk)

        # Online softmax update
        # m_blk: (B, Tq)
        m_blk = np.max(S_blk, axis=-1)

        # m_new: (B, Tq)
        m_new = np.maximum(m, m_blk)

        # exp(m_old - m_new) factor
        alpha = np.exp(m - m_new)  # (B, Tq), note: exp(-inf)=0 ok

        # exp(scores - m_new) for block
        # TODO: compute P_blk = exp(S_blk - m_new[..., None]) safely
        P_blk = None  # TODO  shape (B, Tq, Bk)

        # l_new = alpha*l_old + sum(P_blk)
        l_new = alpha * l + np.sum(P_blk, axis=-1)

        # Update o:
        # term_old = (alpha*l_old)/l_new * o_old
        # term_blk = (P_blk @ V_blk)/l_new
        # TODO: implement both terms
        o = None  # TODO

        # Commit
        m, l = m_new, l_new

    return o


# ============================================================
# Reference implementation (allowed to be slow/memory heavy)
# ============================================================

def reference_attention(Q, K, V, causal: bool = False):
    """
    Full attention reference: softmax(QK^T/sqrt(D) + mask) V
    """
    B, Tq, D = Q.shape
    _, Tk, _ = K.shape
    scale = 1.0 / math.sqrt(D)

    # scores: (B, Tq, Tk)
    S = np.matmul(Q, np.transpose(K, (0, 2, 1))) * scale

    if causal:
        # mask upper triangle: keys > query idx
        key_idx = np.arange(Tk)[None, :]
        q_idx = np.arange(Tq)[:, None]
        mask = key_idx > q_idx  # (Tq, Tk)
        S = np.where(mask[None, :, :], -np.inf, S)

    # stable softmax
    m = np.max(S, axis=-1, keepdims=True)
    P = np.exp(S - m)
    Z = np.sum(P, axis=-1, keepdims=True)
    P = P / Z

    O = np.matmul(P, V)
    return O


# ============================================================
# Tests
# ============================================================

def _assert_allclose(name, a, b, atol, rtol):
    max_abs = np.max(np.abs(a - b))
    denom = np.maximum(1.0, np.max(np.abs(b)))
    rel = max_abs / denom
    if not (np.allclose(a, b, atol=atol, rtol=rtol)):
        raise AssertionError(
            f"{name} failed: max_abs={max_abs:.3e}, rel={rel:.3e}, atol={atol}, rtol={rtol}"
        )

def run_tests():
    rng = np.random.default_rng(0)

    # A few sizes including edge cases
    cases = [
        (1, 1, 1, 8, 8, 16),
        (2, 4, 4, 32, 32, 16),
        (2, 7, 11, 64, 64, 32),
        (1, 16, 16, 128, 128, 64),
    ]

    for (B, Tq, Tk, D, D2, Dv) in cases:
        assert D == D2
        for causal in [False, True]:
            for block_k in [1, 3, 8, 16, 64]:
                Q = rng.standard_normal((B, Tq, D), dtype=np.float64)
                K = rng.standard_normal((B, Tk, D), dtype=np.float64)
                V = rng.standard_normal((B, Tk, Dv), dtype=np.float64)

                ref = reference_attention(Q, K, V, causal=causal)
                out = online_attention(Q, K, V, block_k=block_k, causal=causal)

                # Tolerances: online + exp should match closely in float64
                _assert_allclose(
                    name=f"B={B},Tq={Tq},Tk={Tk},D={D},Dv={Dv},causal={causal},block_k={block_k}",
                    a=out,
                    b=ref,
                    atol=1e-9,
                    rtol=1e-9,
                )

    # Stress numerics: large magnitude logits
    B, Tq, Tk, D, Dv = 2, 8, 33, 64, 16
    Q = rng.standard_normal((B, Tq, D), dtype=np.float64) * 50.0
    K = rng.standard_normal((B, Tk, D), dtype=np.float64) * 50.0
    V = rng.standard_normal((B, Tk, Dv), dtype=np.float64)

    for causal in [False, True]:
        ref = reference_attention(Q, K, V, causal=causal)
        out = online_attention(Q, K, V, block_k=7, causal=causal)
        _assert_allclose("stress_numerics", out, ref, atol=1e-8, rtol=1e-8)

    print("All tests passed ✅")

if __name__ == "__main__":
    run_tests()