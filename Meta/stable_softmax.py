import torch

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
    """
    logits: (..., N)
    mask: same shape as logits, bool, where True means keep and False means mask out
    returns: same shape, softmax along dim
    """
    if mask is None:
        shifted = logits - logits.amax(dim=dim,keepdim=True)
        exps = torch.exp(shifted)
        return exps / exps.sum(dim=dim,keepdim=True)
    
    # make neg_inf values based on mask
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.masked_fill(logits,~mask,neg_inf)
    
    # Check for rows where every value is masked so we can return all zeros there
    all_masked = (~mask).all(dim=dim,keepdim=True)
    
    # avoid subtracting -inf from -inf in fully masked rows
    # if a row is all neg_inf then we make that row all zeros
    # else, the value is unchanged
    safe_logits = torch.where(all_masked, torch.zeros_like(masked), masked)
    
    # now it's safe to shift
    shifted = safe_logits - safe_logits.amax(dim=dim,keepdim=True)
    exps = torch.exp(shifted) * mask.to(logits.dtype)
    
    # calculate denominator
    denom = exps.sum(dim=dim, keepdim=True)
    # if our denominator isn't greater than zero we don't want to blow up
    # our output so we just set final value to zero
    out = torch.where(denom > 0, exps / denom, torch.zeros_like(exps))
    return out


def test_masked_softmax():
    torch.manual_seed(0)

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [1000.0, 1001.0, 1002.0],
    ])
    mask = torch.tensor([
        [True, True, False],
        [False, True, True],
    ])

    out = masked_softmax(logits, mask, dim=-1)

    assert out.shape == logits.shape
    assert torch.allclose(out[0], torch.tensor([0.2689, 0.7311, 0.0000]), atol=1e-4)
    assert torch.allclose(out[1], torch.tensor([0.0000, 0.2689, 0.7311]), atol=1e-4)

    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    logits2 = torch.tensor([[1.0, 2.0]])
    mask2 = torch.tensor([[False, False]])
    out2 = masked_softmax(logits2, mask2)
    assert torch.allclose(out2, torch.zeros_like(out2), atol=1e-6)

    print("test_masked_softmax passed")

test_masked_softmax()