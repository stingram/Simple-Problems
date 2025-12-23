import torch
from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

# dQ/da = 9a^2
# dQ/db = -2b

# dQ/dQ = 1
# since a and b are both 1 by 2, we need to pass 1 by 2 into gradient
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

# if you have a vector valued function y = f(x) then the gradient 
# of y with respect to x is a Jacobian matrix J

# j = (dy/dx1 dy/dx2 ... dy/dxn) = (dy1/dx1 ... dy1/dxn)
#                                  (dy2/dx1 ... dy2/dxn)
#                                  (...                )
#                                  (dym/dx1 ... dym/dxn)
#                                   (mxn)

# torch.autograd is an engine for computing vector-Jacobian product. That
# is given any vector v, it computes J.T*v

# If v happens to be the gradient of a scalar function l = g(y):
#                       v = (dl/dy1 ... dl/dym).T then by the chain
# rul, the vector-Jacobian product would be the gradient of l w.r.t
# x:
# J.T*v =  (dy1/dx1 ... dym/dxn) (dl/dy1)   (dl/dx1)
#          (dy1/dx1 ... dym/dxn) (dl/dy2) = (dl/dx2)
#          (...                ) ( ...  )   ( ...  )
#          (dy1/dxn ... dym/dxn) (dl/dym)   (dl/dxn)
#           (n x m)            * (m x 1) =  (nx1)

# this characteristic of vector-jacobian product is what we use in the above
# example; external_grad represents v

# conceptually, autograd keeps a record of tensors and all executed ops
# in a DAG consisting of Function objects. Leaves are input tensors, roots
# are output tensors. By tracing from roots to leaves, you can automatically
# compute gradients using the chain rule

# in forward pass, autograd
# runs requested to op to compute resulting tensor and
# maintain the operation's gradient function in the DAG

# backward pass happens when .backward() is called on the DAG root
# autograd then:
# copmutes gradients from each .grad_fn
# accumulates them in respective tensor's .grad attribute,
# using chain rule, propagates all the way to the leaf tensors

# dags are dynamic in pytorch - after each .backward() call,
# autograd starts populating a new graph. This allows you to use
# control flow statements in your model; you can change size, shape,
# and ops at every iteration if needed.


# for tensors that don't require gradients, you can set the
# tensors' "required_grad" member to False to exclude if from the
# gradient computation DG

# the output tensor of an operation will require gradients even if
# only a single input tensor has requires_grad=True

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients?: {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")


model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
    
# replace last layer
model.fc = nn.Linear(512,10)

# optimize only the classifier
optimizer = optim.SGD(model.parameters(),lr=1e-2,momentum=0.9)