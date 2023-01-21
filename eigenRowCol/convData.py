import torch
import torch.nn as nn
data = torch.arange(36,dtype=torch.double).reshape(2,2,3,3)
m = nn.Conv2d(2,3,3,stride=1,padding=(1,1),bias=False)
m.weight = nn.Parameter(torch.ones((3,2,3,3)),requires_grad=False)
m.weight.data = m.weight.double()
output = m(data)
output.long()