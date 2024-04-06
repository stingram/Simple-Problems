# reference https://www.mishalaskin.com/posts/tensor_parallel
import numpy as np

def split_columnwise(A, num_splits):
    return np.split(A,num_splits,axis=1)

def split_rowwise(A,num_splits):
    return np.split(A,num_splits,axis=0)

def normal_forward_pass(X,A,B,f):
    Y = f(np.dot(X,A))
    Z = np.dot(Y,B)
    return  Z

def tensor_parallel_forward_pass(X,A,B,f):
    A1,A2=split_columnwise(A,2)
    B1,B2=split_rowwise(B,2)
    Y1=f(np.dot(X,A1))
    Y2=f(np.dot(X,A2))
    Z1=np.dot(Y1,B1)
    Z2=np.dot(Y2,B2)
    Z = Z1 + Z2
    #Z = np.sum([Z1,Z2],axis=0)
    return Z

X = np.random.randn(2,2)
A = np.random.randn(2,2)
B = np.random.randn(2,2)

Z = tensor_parallel_forward_pass(X,A,B, np.tanh)
Z_normal = normal_forward_pass(X,A,B,np.tanh)
print(f"{np.allclose(Z,Z_normal)}")


target = np.array([[-0.5,0.5],[-0.5,0.5]])

# loss function
def L(X,Y):
    return np.sum((Z-Y)**2)
loss = L(Z,target)

def normal_backward_pass(X,A,B,f):
    # recompute forward pas to get activations
    Y = f(np.dot(X,A))
    Z = np.dot(Y,B)
    
    # compute gradient of loss w.r.t Z
    dLdZ = 2*(Z-Y)
    
    # gradient of loss w.r.t B
    # dLdB = dLdZ*dZdB = dLdZ*Y= np.dot(Y.T, dLdZ)  
    dLdB = np.dot(Y.T*dLdZ)