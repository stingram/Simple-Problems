from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm

import warnings

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from model import build_transformer
from dataset import BilingualDataset, casual_mask
from config import get_weights_file_path, get_config

def greedy_decode(model, source, source_mask, tokenizer_src,tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # precopmute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    
    # initializie the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        # build mask for the target
        decoder_mask =  casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # calculate the output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # get the next token
        # we only the last token 
        prob = model.project(out[:,-1])
        
        # select token with the max probability (because this is a greedy search)
        _, next_word = torch.max(prob,dim=1)
        
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)],dim=1)
    
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0) # remove batch dimension
    

def run_validation(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    
    # inference two sentences
    count = 0
    # source_texts = []
    # expected = []
    # predicted = []
    
    # size of control window 
    console_width = 80
    
    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_out = greedy_decode(model,encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)
            
            # print to console
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break
        
    if writer:
        # see on github
        # TorchMetrics CharErrorRate, BLEU, WordErrorRate
            pass

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]


def get_or_build_tokenizer(config, dataset, language):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        
        # for a word to appear in our vocabulary we have to seen it at least twice
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]", "[SOS]", "[EOS]"], min_frequency=2)        
        tokenizer.train_from_iterator(get_all_sentences(dataset, language),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    dataset_raw = load_dataset('opus_books',f"{config['language_source']}-{config['language_target']}", split='train')
    
    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['language_source'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['language_target'])
    
    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9*len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])
    
    train_dataset = BilingualDataset(train_ds_raw, tokenizer_src,
                                      tokenizer_tgt, config['language_source'], config['language_target'],
                                      config['seq_len'])
    val_dataset = BilingualDataset(val_ds_raw, tokenizer_src,
                                      tokenizer_tgt, config['language_source'], config['language_target'],
                                      config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['language_source']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['language_target']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
        
    
    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of taret sentence: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1,shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model
    
    
def train_model(config):
    # define the device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}.")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=config['lr'])
    
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}.")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}...")
        for batch in batch_iterator:
            model.train()
            
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len), hides only padding
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len), hides subsequent words and padding
            
            # run tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # output is (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # output is (batch, seq_len, d_model)
            projection_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)
            
            label = batch['label'].to(device) # (batch, seq_len)
            
            
            # (batch, seq_len, tgt_vocab_size) -> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            # log loss to tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            
            # back prop
            loss.backward()
            
            # update weights
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'],device,lambda msg: batch_iterator.write(msg), global_step, writer)            
        
        # save the model
        model_filename = get_weights_file_path(config, f'{epoch:02d}') 
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename
        )
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)