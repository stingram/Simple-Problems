import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_epoch_grad_accum(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    microbatch_size: int,
):
    """
    x: (N, D)
    y: (N,)
    Perform one epoch over x/y in microbatches.
    Use cross-entropy loss.
    Accumulate gradients across all microbatches, then do one optimizer step.
    Return scalar loss averaged over all examples.
    """
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
    optimizer.zero_grad()
    steps = y.shape[0] // microbatch_size
    losses = torch.zeros(steps)
    for i in range(steps):
        start = i*microbatch_size
        stop = (i+1)*microbatch_size
        pred = model(x[start:stop])
        loss = loss_fn(pred,y[start:stop])
        loss.backward()
        losses[i] = loss
    optimizer.step()
    optimizer.zero_grad()
    return torch.mean(losses).item()
         

def test_train_one_epoch_grad_accum():
    torch.manual_seed(0)
    N, D, C = 16, 10, 4
    x = torch.randn(N, D)
    y = torch.randint(0, C, (N,))

    model = nn.Linear(D, C)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    w_before = model.weight.detach().clone()
    b_before = model.bias.detach().clone()

    loss = train_one_epoch_grad_accum(model, optimizer, x, y, microbatch_size=4)

    assert isinstance(loss, float)
    assert not torch.allclose(model.weight.detach(), w_before)
    assert not torch.allclose(model.bias.detach(), b_before)

    print("test_train_one_epoch_grad_accum passed")

test_train_one_epoch_grad_accum()