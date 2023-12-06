import torch
import torch.nn as nn
from math import sqrt, log

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x)*sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    # seq_len is max length supported
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # need matrix size seq_len, d_model
        # we do this so we only need to calculate this matrix one time
        pe = torch.zeros(seq_len, d_model)
        
        # can use logs for numerical stability
        
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) # (seq_len,1)
        div_term = torch.exp(torch.arange(0,d_model, 2).float() * (-log(10000.0) / d_model) )
        
        # apply sin to even positions
        pe[:,0::2] = torch.sin(position*div_term)
        
        # apply cos to odd positions
        pe[:,1::2] = torch.cos(position*div_term)
        
        # so far this for a single batch so we another dimension for batch
        pe = pe.unsqueeze(0) # -> (1,seq_len,d_model)
        
        # tensor that doesn't have learned weights but you want saved with the weights file
        # (state_dict)
        self.register_buffer('pe', pe)
        
    def forward(self,x):
        # only add x's length values, we don't want to add 
        # positional encodings for words that don't exist in x 
        # because number of words in x will be <= seq_len
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # apply dropout
        return self.dropout(x)
        
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiply
        self.bias = nn.Parameter(torch.zeros(1)) # add
        
    def forward(self, x):
        # want mean of each dimension
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        
        return self.alpha* (x-mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
        
    def forward(self,x):
        # (batch, seq_len, d_model) ---> (Batch,seq_len,d_ff) --> (batch, seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
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
        
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    # sublayer is the previous layer
    # could be either multi-head attention or feedforward
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # need two residual connections
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range (2)])
        
    def forward(self, x, src_mask):
        # why do we need lambda  for the first one but not the second one?
        # TODO - test what happens if I try
        # TODO - x = self.residual_connections[0](x, self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self,x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # need two residual connections
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range (3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # why do we need lambda  for the first one but not the second one?
        # TODO - test what happens if I try
        # TODO - x = self.residual_connections[0](x, self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self,x, encoder_output,src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask, tgt_mask)
        return self.norm(x)
        
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
        
    def forward(self,x):
        # (batch, seq_len, d_model) -> (batch,seq_len, vocab_size)
        return torch.log_softmax(self.proj(x),dim=-1)
    
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 # have two embeddings because this will be a translator (two difference languages)
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 proj: ProjectionLayer):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj
        
    def encode(self, src, src_mask):
        # apply embedding
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6, h: int = 8,
                      dropout: float = 0.1, d_ff: int = 2048):
    # create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # pos 
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # create the transformer
    transformer = Transformer(encoder, decoder,
                              src_embed, tgt_embed,
                              src_pos, tgt_pos,
                              projection_layer)
    
    # use Xavier initialization to reduce training time 
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer

