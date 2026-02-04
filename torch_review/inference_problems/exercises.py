#!/usr/bin/env python3
"""
neuron_numpy_torch_drills.py

Run:
  python neuron_numpy_torch_drills.py
or:
  python -m unittest neuron_numpy_torch_drills.py -v

Goal:
  Implement foundational NumPy/Torch ops that commonly appear in inference enablement / acceleration work.
  Each TODO is designed so you practice:
    - numerically stable reductions (logsumexp, softmax)
    - normalization layers (LayerNorm, RMSNorm)
    - attention (masking, stable softmax, scaling)
    - rotary embeddings (RoPE apply)
    - KV-cache append + gather
    - per-tensor asymmetric quantization

Constraints:
  - Avoid using torch.nn.functional.* in the TODO implementations unless explicitly allowed.
  - Prefer vectorized implementations (no Python loops over large dims).
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import torch
except ImportError as e:
    raise RuntimeError("This script requires PyTorch installed.") from e


# ---------------------------
# Utilities
# ---------------------------

def _set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _allclose(a, b, atol=1e-5, rtol=1e-5) -> bool:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    return np.allclose(a, b, atol=atol, rtol=rtol)


# ============================================================
# EXERCISE 1: NumPy logsumexp (numerically stable)
# ============================================================

def logsumexp_np(x: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute log(sum(exp(x))) stably.

    TODO:
      Implement the standard stabilization trick:
        m = max(x)
        logsumexp = m + log(sum(exp(x-m)))

    Notes:
      - axis can be None (reduce all dims) or an int.
      - must support keepdims.
    """
    # TODO: implement
    m = x.max(axis=axis,keepdims=keepdims)
    return m + np.log(np.sum(np.exp(x-m),axis=axis,keepdims=keepdims))

# ============================================================
# EXERCISE 2: NumPy stable softmax
# ============================================================

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Stable softmax along a given axis.

    TODO:
      Implement softmax using stabilization:
        z = x - max(x)
        expz = exp(z)
        expz / sum(expz)

    Must handle large magnitudes without overflow.
    """
    # TODO: implement
    xmax = logsumexp_np(x,axis,True)
    z =   x - np.array(xmax)
    expz = np.exp(z)
    return np.divide(expz,np.sum(expz,axis=axis,keepdims=True))


# ============================================================
# EXERCISE 3: Torch stable softmax (no torch.softmax)
# ============================================================

def softmax_torch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Stable softmax using torch ops, but do NOT call torch.softmax.

    TODO:
      mirror the numpy approach using torch.max, torch.exp, torch.sum.
    """
    # TODO: implement
    xmax = x.max(dim=dim,keepdim=True)[0]
    z = x - xmax
    expz = torch.exp(z)
    return torch.divide(expz,torch.sum(expz,dim=dim,keepdim=True))


# ============================================================
# EXERCISE 4: Torch LayerNorm forward (no torch.nn.functional.layer_norm)
# ============================================================

def layernorm_torch(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    LayerNorm over the last dimension.

    x: (..., D)
    weight, bias: (D,)

    y = (x - mean) / sqrt(var + eps) * weight + bias

    TODO:
      - compute mean/var over last dim
      - keep broadcasting correct
    """
    # TODO: implement
    xmean = torch.mean(x,dim=-1,keepdim=True)
    xvar = torch.var(x,dim=-1,keepdim=True, unbiased=False)
    xnorm = (x - xmean) / torch.sqrt(xvar+eps)
    return xnorm * weight + bias


# ============================================================
# EXERCISE 5: Torch RMSNorm forward
# ============================================================

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm over last dim:
      rms = sqrt(mean(x^2) + eps)
      y = x / rms * weight

    TODO:
      - compute mean(x^2) over last dim
      - avoid numerical issues
    """
    # TODO: implement
    raise NotImplementedError


# ============================================================
# EXERCISE 6: Scaled dot-product attention (Torch)
# ============================================================

def sdp_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Scaled dot-product attention.

    Shapes (common):
      q, k, v: (B, H, T, D)
      output: (B, H, T, D)

    Compute:
      scores = (q @ k^T) / sqrt(D)
      if attn_mask: add it (mask should be additive: 0 for keep, -inf for mask)
      if causal: apply upper-triangular mask (prevent attending to future)
      p = softmax(scores, dim=-1)
      out = p @ v

    TODO:
      - implement with stable softmax (you may call your softmax_torch)
      - support either/both masks
      - do NOT call torch.nn.functional.scaled_dot_product_attention
    """
    # TODO: implement
    raise NotImplementedError


# ============================================================
# EXERCISE 7: RoPE (Rotary Positional Embedding) apply
# ============================================================

def rope_apply_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to the last dimension, assuming even head dim.

    x: (B, H, T, D) where D is even
    cos, sin: broadcastable to (B, H, T, D/2) or (1, 1, T, D/2)

    Standard RoPE:
      split x into (x_even, x_odd) pairs:
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
      rotate:
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
      interleave back into D

    TODO:
      - implement rotation + interleave without loops
    """
    # TODO: implement
    raise NotImplementedError


def rope_build_cos_sin_torch(T: int, D: int, base: float = 10000.0, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cos/sin tables for RoPE.

    cos, sin shapes: (1, 1, T, D/2)

    This is provided (not a TODO), so tests can focus on rope_apply_torch.
    """
    assert D % 2 == 0
    half = D // 2
    # inv_freq: (half,)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(T, device=device, dtype=torch.float32)  # (T,)
    freqs = torch.einsum("t,f->tf", t, inv_freq)  # (T, half)
    cos = torch.cos(freqs)[None, None, :, :]  # (1,1,T,half)
    sin = torch.sin(freqs)[None, None, :, :]
    return cos, sin


# ============================================================
# EXERCISE 8: KV-cache append + gather
# ============================================================

@dataclass
class KVCache:
    """
    Simple KV cache for a single layer, per batch & head.

    k, v: (B, H, T_max, D)
    cur: current length (int)
    """
    k: torch.Tensor
    v: torch.Tensor
    cur: int

def kv_cache_init(B: int, H: int, T_max: int, D: int, device: str = "cpu") -> KVCache:
    k = torch.empty((B, H, T_max, D), device=device, dtype=torch.float32)
    v = torch.empty((B, H, T_max, D), device=device, dtype=torch.float32)
    return KVCache(k=k, v=v, cur=0)

def kv_cache_append(cache: KVCache, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
    """
    Append new tokens into the cache.

    k_new, v_new: (B, H, T_new, D)

    TODO:
      - write into cache.k/cache.v starting at cache.cur
      - update cache.cur
      - validate no overflow
    """
    # TODO: implement
    raise NotImplementedError

def kv_cache_gather(cache: KVCache, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather a subset of time positions from the cache.

    idx: (B, T_sel) int64, indices into [0, cache.cur)

    Return:
      k_sel, v_sel: (B, H, T_sel, D)

    TODO:
      - gather along time dimension
      - must be batch-aware (each batch can have different idx values)
      - no Python loops over B
    """
    # TODO: implement
    raise NotImplementedError


# ============================================================
# EXERCISE 9: Per-tensor asymmetric quantize/dequantize (NumPy)
# ============================================================

def quantize_dequantize_int8_np(x: np.ndarray, qmin: int = -128, qmax: int = 127) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """
    Per-tensor asymmetric quantization.

    Given float x, compute:
      scale s, zero-point z
      q = clip(round(x/s) + z, qmin, qmax)
      x_hat = s * (q - z)

    Return:
      q_int8 (np.int8), scale (float), zero_point (int), x_hat (float array)

    TODO:
      - choose scale and zero-point using min/max mapping to [qmin, qmax]
      - handle degenerate case where x_min == x_max
    """
    # TODO: implement
    raise NotImplementedError


# ============================================================
# EXERCISE 10: Top-p (nucleus) filtering for logits (Torch)
# ============================================================

def top_p_filtering_torch(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Nucleus (top-p) filtering.

    logits: (B, V)
    Return new logits where tokens outside nucleus are set to -inf.

    Steps:
      1) sort logits descending -> sorted_logits, sorted_idx
      2) probs = softmax(sorted_logits)
      3) cumulative = cumsum(probs)
      4) mask tokens where cumulative > p (but keep at least 1 token)
      5) scatter mask back to original order, set masked logits to -inf

    TODO:
      - implement with torch ops
      - you may call your softmax_torch
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------
# Reference implementations (for tests only)
# ---------------------------

def _softmax_torch_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)

def _layernorm_torch_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # reference using basic ops (not F.layer_norm), so it's comparable to your implementation
    mu = x.mean(dim=-1, keepdim=True)
    var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
    y = (x - mu) / torch.sqrt(var + eps)
    return y * weight + bias

def _rmsnorm_torch_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * weight

def _sdp_attention_torch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], causal: bool) -> torch.Tensor:
    D = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    if causal:
        T = q.shape[-2]
        causal_mask = torch.triu(torch.ones((T, T), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))
    if attn_mask is not None:
        scores = scores + attn_mask  # additive mask
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)

def _rope_apply_torch_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    y = torch.empty_like(x)
    y[..., 0::2] = y0
    y[..., 1::2] = y1
    return y

def _top_p_filtering_ref(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    # mask tokens where cum > p, but keep first token always
    mask_sorted = cum > p
    mask_sorted[..., 0] = False
    # shift mask right so we keep the first token that crosses p
    # (common convention is: remove tokens AFTER threshold; this achieves that)
    mask_sorted[..., 1:] = mask_sorted[..., :-1].clone()
    mask_sorted[..., 0] = False
    # scatter back
    mask = torch.zeros_like(mask_sorted).scatter(-1, sorted_idx, mask_sorted)
    out = logits.clone()
    out[mask] = float("-inf")
    return out


# ---------------------------
# Tests
# ---------------------------

class TestNeuronDrills(unittest.TestCase):
    def setUp(self) -> None:
        _set_seeds(0)

    def test_logsumexp_np(self):
        x = np.random.randn(3, 5).astype(np.float64) * 50.0  # stress stability
        got = logsumexp_np(x, axis=1, keepdims=True)
        # reference via stable numpy
        m = np.max(x, axis=1, keepdims=True)
        ref = m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))
        self.assertTrue(np.allclose(got, ref, atol=1e-10, rtol=1e-10))

    def test_softmax_np(self):
        x = (np.random.randn(4, 7) * 80.0).astype(np.float64)
        got = softmax_np(x, axis=1)
        ref = np.exp(x - np.max(x, axis=1, keepdims=True))
        ref = ref / np.sum(ref, axis=1, keepdims=True)
        self.assertTrue(np.allclose(got, ref, atol=1e-10, rtol=1e-10))
        self.assertTrue(np.allclose(got.sum(axis=1), 1.0, atol=1e-10))

    def test_softmax_torch(self):
        x = torch.randn(5, 11, dtype=torch.float32) * 60.0
        got = softmax_torch(x, dim=-1)
        ref = _softmax_torch_ref(x, dim=-1)
        self.assertTrue(_allclose(got, ref, atol=1e-5, rtol=1e-5))
        self.assertTrue(_allclose(got.sum(dim=-1), torch.ones(5), atol=1e-5, rtol=1e-5))

    def test_layernorm_torch(self):
        B, T, D = 2, 3, 16
        x = torch.randn(B, T, D)
        w = torch.randn(D)
        b = torch.randn(D)
        got = layernorm_torch(x, w, b, eps=1e-5)
        ref = _layernorm_torch_ref(x, w, b, eps=1e-5)
        self.assertTrue(_allclose(got, ref, atol=1e-5, rtol=1e-5))

    def test_rmsnorm_torch(self):
        B, T, D = 2, 4, 32
        x = torch.randn(B, T, D)
        w = torch.randn(D)
        got = rmsnorm_torch(x, w, eps=1e-6)
        ref = _rmsnorm_torch_ref(x, w, eps=1e-6)
        self.assertTrue(_allclose(got, ref, atol=1e-5, rtol=1e-5))

    def test_sdp_attention_torch_no_mask(self):
        B, H, T, D = 2, 3, 5, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        got = sdp_attention_torch(q, k, v, attn_mask=None, causal=False)
        ref = _sdp_attention_torch_ref(q, k, v, attn_mask=None, causal=False)
        self.assertTrue(_allclose(got, ref, atol=1e-4, rtol=1e-4))

    def test_sdp_attention_torch_causal(self):
        B, H, T, D = 1, 2, 6, 32
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        got = sdp_attention_torch(q, k, v, attn_mask=None, causal=True)
        ref = _sdp_attention_torch_ref(q, k, v, attn_mask=None, causal=True)
        self.assertTrue(_allclose(got, ref, atol=1e-4, rtol=1e-4))

    def test_sdp_attention_torch_additive_mask(self):
        B, H, T, D = 2, 2, 4, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        # Mask out last key position for all queries
        mask = torch.zeros(B, 1, 1, T)
        mask[..., -1] = float("-inf")
        got = sdp_attention_torch(q, k, v, attn_mask=mask, causal=False)
        ref = _sdp_attention_torch_ref(q, k, v, attn_mask=mask, causal=False)
        self.assertTrue(_allclose(got, ref, atol=1e-4, rtol=1e-4))

    def test_rope_apply_torch(self):
        B, H, T, D = 2, 3, 7, 32
        x = torch.randn(B, H, T, D)
        cos, sin = rope_build_cos_sin_torch(T=T, D=D, device="cpu")
        got = rope_apply_torch(x, cos, sin)
        ref = _rope_apply_torch_ref(x, cos, sin)
        self.assertTrue(_allclose(got, ref, atol=1e-5, rtol=1e-5))

    def test_kv_cache_append_and_gather(self):
        B, H, T_max, D = 2, 2, 16, 8
        cache = kv_cache_init(B, H, T_max, D)
        # append 5, then 3 tokens
        k1 = torch.randn(B, H, 5, D)
        v1 = torch.randn(B, H, 5, D)
        kv_cache_append(cache, k1, v1)
        self.assertEqual(cache.cur, 5)
        k2 = torch.randn(B, H, 3, D)
        v2 = torch.randn(B, H, 3, D)
        kv_cache_append(cache, k2, v2)
        self.assertEqual(cache.cur, 8)

        # build reference full
        k_full = torch.cat([k1, k2], dim=2)
        v_full = torch.cat([v1, v2], dim=2)

        idx = torch.tensor([[0, 3, 7], [1, 2, 6]], dtype=torch.int64)  # (B, T_sel)
        k_sel, v_sel = kv_cache_gather(cache, idx)

        # reference gather
        # fancy gather: expand idx to (B, H, T_sel, D)
        idx_exp = idx[:, None, :, None].expand(B, H, idx.shape[1], D)
        k_ref = torch.gather(k_full, dim=2, index=idx_exp)
        v_ref = torch.gather(v_full, dim=2, index=idx_exp)

        self.assertTrue(_allclose(k_sel, k_ref, atol=1e-5, rtol=1e-5))
        self.assertTrue(_allclose(v_sel, v_ref, atol=1e-5, rtol=1e-5))

    def test_quantize_dequantize_int8_np(self):
        x = (np.random.randn(1024).astype(np.float32) * 3.0) + 0.2
        q, s, z, xhat = quantize_dequantize_int8_np(x)
        self.assertEqual(q.dtype, np.int8)
        self.assertTrue(np.isfinite(s) and s > 0.0)
        self.assertTrue(isinstance(z, int))
        # sanity: dequant should be close-ish
        mse = np.mean((x - xhat) ** 2)
        self.assertLess(mse, 0.05)  # loose; depends on distribution but should pass for typical gaussian

    def test_top_p_filtering(self):
        B, V = 3, 50
        logits = torch.randn(B, V) * 2.0
        got = top_p_filtering_torch(logits, p=0.9)
        ref = _top_p_filtering_ref(logits, p=0.9)
        # exact match for -inf mask positions + close elsewhere
        self.assertTrue(torch.equal(torch.isinf(got), torch.isinf(ref)))
        # compare finite entries
        mask = ~torch.isinf(ref)
        self.assertTrue(_allclose(got[mask], ref[mask], atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
