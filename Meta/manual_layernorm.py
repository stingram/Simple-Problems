import torch

def manual_layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    x: (..., D)
    gamma: (D,)
    beta: (D,)
    normalize over last dimension
    """
    xmean = torch.mean(x,dim=-1,keepdim=True)
    xvar = torch.var(x,dim=-1,keepdim=True, unbiased=False)
    xhat= (x - xmean)/torch.sqrt(xvar+eps)
    return xhat * gamma + beta
    
    


def test_manual_layernorm():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 5)
    gamma = torch.randn(5)
    beta = torch.randn(5)

    out = manual_layernorm(x, gamma, beta)

    ref = torch.nn.functional.layer_norm(x, normalized_shape=(5,), weight=gamma, bias=beta)
    assert out.shape == x.shape
    assert torch.allclose(out, ref, atol=1e-5)
    print("test_manual_layernorm passed")
    # if not torch.allclose(out, ref, atol=1e-5):
    #     print(f'{out=}')
    #     print(f'{ref=}')
    #     raise AssertionError
    # else:
    #     print("test_manual_layernorm passed")

test_manual_layernorm()