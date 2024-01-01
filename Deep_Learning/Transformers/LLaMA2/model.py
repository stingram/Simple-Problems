import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int =4096
    n_layers: int = 32
    n_heads: int = 32 # number of heads for the queries
    n_kv_heads: Optional[int] = None # number of heads for K and V
    vocab_size: int = -1 # set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    # needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in Roformer paper, the dimension of the embedding must be even
    assert head_dim % 2 == 0, "Dimensions must be divisible by 2"
    
    # build the theta parameters
    # according to the formula theta_i = 10000 ^ (-2(i-1)/dim) for i = [1,2,..., dim/2]
    # shape: (Head_dim /2)
    theta_numerator = torch.arange(0, head_dim,2).float()
    
    # shape: (head_dim /2)
    theta = 1.0 / (theta ** (theta_numerator/ head_dim)).to(device)

    # construct the positions (the m parameter)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    
    # multiply each theta by each position using the outer product
    # shape: (seq_len) *outer_product* (head_dim / 2) -> (seq_len, head_dim /2)
    freqs = torch.outer(m,theta).float()
    
    # we can copmute complex numbers in the polar form c = R*exp(i *m * theta), where R =1 as follows:
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # split embedding and create real and imaginary parts
    # The reshape part basically says take two consecutive dimensions and group them
    
    # (Batch, seq_len, H, Head_dim) -> (Batch, seq_len, H, Head_dim /2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # (seq_len, head_dim /2) -> (1, seq_len, 1, head_dim /2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    # now we do element-wise multiplication of re, im parts of embedding times cos + isin vector (which is a rotation)
    # (Batch, seq_len, H, head_dim /2) * (1, seq_len, 1, Head_dim /2) = (batch, seq_len, H, Head_dim /2)
    x_rotated = x_complex * freqs_complex
    
    # Then we split re/im parts of x_rotated 
    # (batch, seq_len, H, head_dim /2) -> (batch, seq_len, H, head_dim/ 2, 2)
    x_out = torch.view_as_real(x_rotated)
    
    # Reshape into a vector
    # (batch, seq_len, H, Head_dim/ 2, 2) -> (batch, seq_len, H, Head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # (batch, seq_len, N_KV_heads, 1, Head_dim)
        return (
            x[:,:,:, None,:]
            .expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim)
            .reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim)
        )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x: torch.Tensor):
        # (batch, seq_len, dim) * (batch,seq_len,1) =  (batch,seq_len,dim)
        # rsqrt = 1/(sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    
    def forward(self,x: torch.Tensor):
        # (dim) * (batch, seq_len, dim) = (batch, seq_len,dim)
        return self.weight*self._norm(x.float()).type_as(x)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        # non-parallelized version
        
        # indicates number of heads for keys and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # indicates number of heads for the queries
        self.n_heads_q = args.n_heads
        # indicates how many times the keys and values should be repeated to match the head of the queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads*self.head_dim, args.dim, bias=False)
        
        # KV cache
        self.cache_k = torch.zeros((args.max_batch_size,args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size,args.max_seq_len, self.n_kv_heads, self.head_dim))
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (batch, seq_len, dim) =  (batch, 1, dim)
        
        # apply Wq, Wk, and Wv matrices to queries, keys, and values
        
        # (B,1,Dim) -> (B,1, H_Q*Head_dim)
        xq = self.wq(x)
        
        # (B,1,Dim) -> (B,1, H_KV*Head_dim)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # (B, 1, H_Q * Head_dim) -> (B, 1, H_Q,Head_dim)
        xq  = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        
        # (B, 1, H_KV * Head_dim) -> (B, 1, H_KV,Head_dim)
        xk  = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)        
        xv  = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # apply rotary position encodings to queries and keys
        # doesn't change the shape of the tensors
        xq = apply_rotary_embeddings(xq,freqs_complex,device=x.device)
        xk = apply_rotary_embeddings(xk,freqs_complex,device=x.device)
        
        # replace the entry in the cacher for this token
        self.cache_k[:batch_size,start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size,start_pos:start_pos+seq_len] = xv

        # retrieve all the cached keys and values so far
        # (B, seq_len_KV, H_KV, Head_dim)
        keys = self.cache_k[:batch_size,0:start_pos+seq_len]
        values = self.cache_v[:batch_size,0:start_pos+seq_len]

        # Replicate the heads of the K and V results to match the number of heads for Q
        keys = repeat_kv(keys,self.n_rep)
        values = repeat_kv(values,self.n_rep)
        
        # now do multi-head attention
        # (B,seq_len=1, H_Q, Head_dim) -> (B, H_Q, seq_len=1, Head_dim)
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # scores
        # (batch, H_Q, seq-len=1, Head_Dim) @ (batch, H_Q, Head_dim, seq_len_Kv) -> (B, H_Q, seq_len=1, seq_len_KV)
        scores = torch.matmul(xq,keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1=seq_len,seq_len_kv) @ (B, H_Q, seq_len_kv, Head_dim) -> (B,H_Q,1=seq_len,head_dim)
        out = torch.matmul(scores,values)
        
        # (B,H_Q,1=seq_len,head_dim) -> (B,1=seq_len,H_Q, head_dim) -> (B,1=seq_len,H_Q*head_dim=dim)
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B,1=seq_len,dim) -> (B,1=seq_len,dim)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2* hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier*hidden_dim)
            
        # round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim+args.multiple_of - 1) // args.multiple_of)
        # hidden_size = 7, multiple of 5
        # (7+4) // 5  = 2
        # 2 * 5 = 10
        
        # 
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # normalization before self-attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # normalization before the feed-foward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (batch, seq_len, dim) + (batch, seq_len, dim) = (batch, seq_len,  
        h = x + self.attention.forward(self.attention_norm(x),start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
 
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.token_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
            
        self.norm = RMSNorm(args.dim, eps=args.eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2,
                                                              device=self.args.device)
        
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # remember that because are using KV-cache, sequence length will always be 1
        
        #(Batch, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"
        
        # (batch, seq_len) -> (batch, seq_len, Embedding_dim)
        h = self.token_embeddings(tokens)
        
        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]
        
        # consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h,start_pos,freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
    
     