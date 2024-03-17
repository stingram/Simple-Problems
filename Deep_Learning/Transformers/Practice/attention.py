import torch
import torch.nn as nn
from math import sqrt, log
   
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        # be careful with number of heads
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model//h
        
        # define matrices for Q,K,V
        self.w_q = nn.Linear(d_model,d_model) # wq
        self.w_k = nn.Linear(d_model,d_model) # wv
        self.w_v = nn.Linear(d_model,d_model) # wv
        
        # output matrix (h*d_k, d_model)
        self.w_o = nn.Linear(d_model,d_model)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query,key,value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # the transpose swaps last two dimensions
        #  key will have shape: (batch, # heads, seq_len, d_k)-> (batch, # heads, d_k, seq_len)
        # so then we'll have attention_scores shape (batch, # heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / sqrt(d_k)
        
        # before softmax we need to mask values
        if mask is not None:
            # we define mask such that if a value in the mask tensor is 0,
            # then in attention_scores we set the value to approx -inf, or -1e9
            attention_scores.masked_fill(mask == 0, -1e9)
            
        # (batch, h, seq_len, seq_len)
        # TODO - check if (0,0,0,:) == 0  or if it's (0,0,:,0) == 0
        # I assume it's the former. 
        # I was correct it is each row that sums to 1 after softmax
        attention_scores = attention_scores.softmax(dim= -1)
        # print(f'sum: {sum(attention_scores[0,0,0,:])}') 
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        # saving tuple for visualization
        return (attention_scores @ value), attention_scores
        
    def forward(self, q,k,v, mask):
        # we need mask to prevent words from interacting 
        
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        
        # split up using view method
        # we swap seq_len with self.h to get -> (batch,# heads, seq_len, d_k)
        # each head will each word but only a subset of the embeddings
        
        # (batch,seq_len, d_model) -> 
        # (batch, seq_len, # heads, d_k) ->
        # (batch,# heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask, self.dropout)
        
        # (batch, h,seq_len,d_k)  -> 
        # (batch, seq_len, h,d_k) ->
        # (batch, seq_len, d_model)
        # contiguous makes sure that in-place, all the elements are viewed as needed 
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x) 
        