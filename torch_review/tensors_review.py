import torch
import numpy as np

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
print(f'Ones Tensor: \n {x_ones} \n')

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f'Random Tensor: \n {x_rand} \n')


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor}\n")
print(f"Ones Tensor: \n {ones_tensor}\n")
print(f"Zeros Tensor: \n {zeros_tensor}\n")

tensor = torch.rand(3,4)
print(f"shape of tensor: {tensor.shape}")
print(f"datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f'Device tensor is stored on: {tensor.device}')
    
tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor,tensor,tensor],dim=1)
print(t1)

# element-wise multiplication
print(f'tensor.mul(tensor)\n {tensor.mul(tensor)}\n')
# alt syntax
print(f'tensor * tensor \n {tensor*tensor}\n')

# matmul
print(f'tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)}\n')
# alt snytax
print(f'tensor @ tensor \n {tensor @ tensor.T}\n')

# operations with "_" suffix are in-place.
# For example x.copy_(y), x.t_() will change x
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# in-places ops save memory,  but can be problematic when computing
# derivatives because of an immediate loss of history.

# tensors on the CPU and numpy arrats can share their underlying memory
# locations and changing one will change the other
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n,1,out=n)
print(f"t: {t}")
print(f"n: {n}")