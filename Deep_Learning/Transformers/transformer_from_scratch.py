# http://peterbloem.nl/blog/transformers

# Lectures: https://www.youtube.com/playlist?list=PLIXJ-Sacf8u60G1TwcznBmK6rEL3gmZmV

from pyclbr import Class
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# create tensor x with size b, t, k
# b = 2, t = 3, k = 4

# using dummy values here, in the real-world we'd have real feature vectors
x = np.random.randn(2,3,4)
X = torch.from_numpy(x)
print(f"X: {X}.")

raw_weights = torch.bmm(X, X.transpose(1,2)) # transpose across t and k dimension, since we are using batch matrix multiply
# since we did X*X_t, we have a symmetric matrix, meaning we have raw_weights = raw_weights_t
# this also means some weights are redundant
# raw_weights maxtrix dimension size is  t by t

print(f"raw_weights: {raw_weights}.")


# Turn raw weighs W'_ij into positive values that sum to one
weights = F.softmax(raw_weights.float(), dim=2)

print(f"weights: {weights}.")

# now we can compute the output sequence by doing batch matrix multiplication of weights and X
y = torch.bmm(weights, X.float())

# Two matrix multiplications and one soft-max gives us basic self-attention.
print(f"y: {y}.")

# Now we want three additional tricks
# 1) Queries, Keys, and Values
# Every input vector x_i is used in three ways
#   It's compared to every other vector to establish the weights for it's own output y_i (Query)
#   It's compared to every other vector to establish the weights for the output of the j-th vector y_j (Key)
#   It is used as part of the weighted sum to compute each output vector once the weights have been established (Value)

# In basic self attention, each input vector must play all three roles described above. Instead, we can apply three 
# k by k weight matrices W_q, W_k, W_v and three linear transformations of each x_i, for the three parts in self-attention
# q_i = W_q*x_i, k_i = W_k*x_i, v_i = W_v*x_i
# Then we have w'_ij = (q_i)_t*k_j, w_ij = softmax(w'_ij), y_i = sum_over_j(w_ij*v_j)  



# 2) Scaling the dot product
# Softmax can be sensitive to very large input values. This sensitivity can kill the gradient, slow learning, or stop learning.
# This is a function of the size of the embedding dimension k, so as k grows, values as input to softmax function also grow.
# We can scale back the dot product by doing:
# w'_ij = ((q_i)_t*k_j)/sqrt(k) 

# Why sqrt(k)? Well, imagine a vector in R_k with values all c. Then it's euclidean length is sqrt(k)*c. Therefore, we are dividing
# out the amount by which the increase in dimension increases the length of the average vectors.

# 3) Multi-head attention
# This is to account for the fact that a word can mean different things to different neighbors.
# In a single self-attention operation, the word order wouldn't change the result, even though it should. 
# y_i would be the same for the same x_i (just in a different position)

# We can deal with this by combining several attention mechanisms (indexed with r), eaech with different
# matrices Wʳ_q, Wʳₖ, Wʳᵥ. These are called attention heads. For input xᵢ, each attention head produces
# a different output vector yʳᵢ. We concatenate these, and then pass them through a linear transformation to
# reduce the dimension back to k.        
  
# Efficient multi-head self-attention - this a small number of copies of the self-attention mechanism applied in parallel,
# each with it's own key, value, and query transformation. Problem is that for R heads,  the self attention operation is
# R times as slow. 

# To make this almost as fast as single-head attention, we cut each incoming vector into chunks. If each input vector
# has 256 dimensions, and we have 8 attention heads, we can cut each vector into 8 chunks of 32 dimensions. For each chunk,
# we generate keys, values, and queries of 32 dimensions eachs. This means that Wʳ_q, Wʳₖ, Wʳᵥ are all 32 by 32.

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k = k
        self.heads = heads
        
        # We can think of the h attention heads as h separate sets of three matrices Wʳ_q, Wʳₖ, Wʳᵥ,
        # but it's actually more efficient to combine all heads into three single h*k by k matrices,
        # so we can compute all the concatenated queries, keys, and values in a single matrix multiplication
        
        # Compute queries, keys, and values for all heads
        self.tokeys = nn.Linear(k, k * heads, bias=False) # becomes one long vector of keys 
        self.toqueries = nn.Linear(k, k * heads, bias=False) # becomes one long vector queries
        self.tovalues = nn.Linear(k, k * heads, bias=False) # becomes one long vector of values
        
        # Reduces feature size from size heads * k to size k
        self.unifyheads = nn.Linear(heads * k, k)
        
    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        
        # since the output of each linear module has size (b,t,h*k) we reshape to
        # give each heade its own dimension
        queries = self.toqueries(x).view(b,t,h,k)
        keys = self.tokeys(x).view(b,t,h,k)
        values = self.tovalues(x).view(b,t,h,k)
        
        # Next we need to compute dot products, this is the same operattion for every head, so we
        # fold heads into the batch dimension. Then we can use torch.bmm() as before, and the whole
        # collection of keys, queries, and values will just be seen as a slightly larger batch
        
        # Unfortunately, we need to transpose before we shape to get head and batch dimension next
        # to each other. Now the dot products can be computed in a single matrix multiplication, but
        # now between queries and keys
        keys = keys.transpose(1,2).contiguous().view(b*h,t,k)
        queries = queries.transpose(1,2).contiguous().view(b*h,t,k)
        values = values.transpose(1,2).contiguous().view(b*h,t,k)
        
        # Instead of scaling the dot product later, we do scaling earlier to save memory.
        # Because we are doing it sooner, we scale queries nad keys by 4th root of k.
        # It's 4th root of k because sqrt(k)*sqrt(k) is 4th root of k
        queries = queries / (k**(1/4))
        keys = keys / (k**(1/4))
        
        # get dot product, dot size is now (b*h,t,t) containing raw weights
        dot = torch.bmm(queries, keys.transpose(1,2))
        
        # softmax, now dot contains row-wise normalized weights
        dot = F.softmax(dot, dim=2)
        
        # apply the self attention to the values and change view to b,h,t,k as before
        out = torch.bmm(dot,values).view(b,h,t,k)
        
        # to unify attention heads, we transpose again so the head dimension and the embedding
        # dimension are next to each other, and we reshape to get concatentated vectors of 
        # dimension k*h. We then pass these through the unifyheads layer to project back down to 
        # k dimensions.
        
        # swap h, t back to order as before, unify heads
        out = out.transpose(1,2).contiguous().view(b,t,h*k)
        return self.unifyheads(out)
        
        
        # What is a transformer?
        # Any architecture designed to process a connected set of units - such as the tokens
        # in a sequence of the pixels in an image - where the only interaction between units
        # is through self-attention.
        
        # For a transformer block, the important thing is to combine self-attention with a local
        # feedforward and to add normalization and residual connections.
        
        # Normalization and residual connections are used to help train faster and more accurately.
        # The layer normalization is applied over the embedding dimension only. 
        
        
# note the completely arbitrary choice of making the hidden layer of the feedforward 4 times
# as big as the input and output. Smaller values may work as well, and save memory, but it
# shoudl be bigger than the input/output layers.
class TransformerBlock(nn.Module):
    def __init__(self, k, heads) -> None:
        super().__init__()
        
        self.attention = SelfAttention(k,heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        
        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k,k)
        )
        
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended+x) # residual
        
        fedforward = self.ff(x)
        return self.norm2(fedforward+x)