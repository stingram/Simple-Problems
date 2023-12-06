import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Any

# With the diagonal as 1, then only items above the diagonal are 1
# https://pytorch.org/docs/stable/generated/torch.triu.html
def casual_mask(seq_size):
    mask = torch.triu(torch.ones(1,seq_size,seq_size),diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt,
                 src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # we need to pad each token list to match seq-Len
        # we subtract two because we'll add SOS and EOS anyway
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        
        # we subtract one because we'll add SOS. The model will produce EOS eventually
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        # Add SOS and EOS to source text (plus padding)
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
        ])
        
        # for decoder input we only want to add SOS token
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ])
        
        # For the label, we don't need SOS token, but we do want EOS token before the padding
        # (what we expect as output from the decoder)
        label = torch.cat([
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ])
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input, # seq_len
            "decoder_input": decoder_input, # seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1, seq_len) & (1,seq_len,seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
