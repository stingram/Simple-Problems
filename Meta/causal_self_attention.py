import math
import torch

def causal_self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q, k, v: (B, T, D)
    returns: (B, T, D)

    Implement:
      scores = q @ k.transpose(-2, -1) / sqrt(D)
      causal mask so position t cannot attend to > t
      probs = softmax(scores)
      out = probs @ v
    """
    scores = q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1])
    T = q.shape[-2]
    mask = torch.triu(torch.ones(T,T),diagonal=1).bool()
    print(f'{mask=}')
    masked = torch.masked_fill(scores,mask,float('-inf'))
    probs = torch.softmax(masked,dim=-1)
    return probs @ v

def test_causal_self_attention():
    torch.manual_seed(0)
    B, T, D = 2, 4, 8
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)

    out = causal_self_attention(q, k, v)
    assert out.shape == (B, T, D)

    # Position 0 should only attend to itself.
    q0 = q[:, :1, :]
    k0 = k[:, :1, :]
    v0 = v[:, :1, :]
    ref0 = torch.softmax((q0 @ k0.transpose(-2, -1)) / math.sqrt(D), dim=-1) @ v0
    assert torch.allclose(out[:, :1, :], ref0, atol=1e-5)

    print("test_causal_self_attention passed")

test_causal_self_attention()