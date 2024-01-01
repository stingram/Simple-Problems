from typing import Optional, List
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
        
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        # Convert to absolute path
        checkpoints_dir = Path(checkpoints_dir).resolve()
        
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, "No checkpoint files found"
            chk_path = checkpoints[0]
            print(f"Loading checkpoint {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu")
            print(f"Loaded checkpoint in {(time.time() - prev_time):.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
            
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
            
        model = Transformer(model_args).to(device)
        
        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint,strict=True)
            print(f"Loaded state dict in {(time.time() - prev_time):.2f}s")
        
        return LLaMA(model,tokenizer,model_args)


    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
            
        # convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        
        # make sure that the batch size isn't too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # make sure the prompt length is not larger than the max seq length
        assert max_prompt_len <= self.args.max_seq_len
        
        total_len = min(self.args.max_seq_len, max_gen_len+max_prompt_len)
        
        # create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # populate the initial tokens with the prompt tokens
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            
        eos_reached = torch.tensor([False]*batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        for cur_pos in tqdm(range(1,total_len), desc='Generate tokens'):
            with torch.no_grad():
                logits = self.model.forward(tokens[:,cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                # the temperature is applied before the softmax
                probs = torch.softmax(logits[:,-1] / temperature, dim = -1)
                next_token = self._sample_top_p(probs,top_p)
            else:
                # greedly select the token with the maximum probability
                next_token = torch.argmax(logits[:,-1],dim =-1)
            
            # flatten next_token
            next_token = next_token.reshape(-1)
            
            # only replace the token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:,cur_pos] = next_token
            
            # EOS is reached only if we find an EOS token for a padding position
            # this eos_reached will be a vector of booleans for each prompt in the batch
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break
            
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # cut to the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)
            
    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs,dim=-1,descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        # redistribute probabilities
        probs_sort.div_(probs_sort.sum(dim=-1,keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token) # TODO - understand this
        return next_token
        
if __name__ == '__main__':
    torch.manual_seed(42)
    
    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    
    model = LLaMA.build(
        checkpoints_dir="llama/llama-2-7b",
        tokenizer_path="llama/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )
    
    print(f"All OK")
    
    # Inference the Model
    # greedy, beam search, temperature, random sampling, top K, top P
    # greedy - select token with maximum probability. easy to implement, poor performance in practice. If something wrong happens earlier, "recovery" is unlikely 
    
    # beam search - parameter K, at every step keep top K performing tokens
    # at next time step we make two prompts, one with each token from before
    # then, cumulative score for next two for each prompt is used to pick which of the two should have
    # been used. Then for these two new tokens, process repeats
    # generally performs better than greedy strategy, but it is slower than greedy
    
    # temperature - scales the logits before softmax,
    # low temperature, makes the model more confident by scaling logits up
    # high temperature, makes the model less confident by scaling logits down
    # can be used in conjunction with other strategies
    
    # random sampling
    # we sample from the random distribution that is output from the softmax
    # with very little probability it may happen that we choose tokens that are total nonsense
    
    # top K - only sample from top k highest probabilities so that low probabilities are never chosen
    # given some distributions, low-probability tokens still make their way into the top k tokens
    # this can be an issue when a distribution is pretty flat or too few are high enough probability
    
    # top P - only sample from probabilities whose cumulative prbability are >= P
    
    # LLaMA implements top P strategy
    
    
    prompts = [
        "Simply put, the theory of relativity states that "
        #TODO - ADD MORE LATER
    ]
    
    out_tokens, out_text = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f"{out_text[i]}")
        print('-'*50)