import math
import torch

def kv_cache_step(
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
):
    """
    k_cache, v_cache: (B, T_prev, D) or None
    q_new, k_new, v_new: (B, 1, D)

    Returns:
      out_new: (B, 1, D)
      new_k_cache: (B, T_prev + 1, D)
      new_v_cache: (B, T_prev + 1, D)
    """
    # always concatenate along T dimension
    if k_cache is not None:
        k_cache = torch.cat([k_cache, k_new],dim=1)
        v_cache = torch.cat([v_cache, v_new],dim=1)
    else:
        k_cache = k_new
        v_cache = v_new
    
    attn = q_new @ k_cache.transpose(-2,-1)
    # shouldn't need a mask at all
    D = k_cache.shape[-1]
    s = torch.softmax(attn / math.sqrt(D), dim=-1)
    res = s @ v_cache
    print(f'{res.shape=}')
    return (res, k_cache, v_cache)

def test_kv_cache_step():
    torch.manual_seed(0)
    B, T, D = 2, 5, 8

    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)

    k_cache, v_cache = None, None
    outs = []

    for t in range(T):
        out_t, k_cache, v_cache = kv_cache_step(
            k_cache, v_cache,
            q[:, t:t+1, :],
            k[:, t:t+1, :],
            v[:, t:t+1, :],
        )
        outs.append(out_t)

    out_decode = torch.cat(outs, dim=1)

    # Reference from full causal attention
    def full_causal(q, k, v):
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return probs @ v

    out_ref = full_causal(q, k, v)
    assert torch.allclose(out_decode, out_ref, atol=1e-5)

    print("test_kv_cache_step passed")

test_kv_cache_step()