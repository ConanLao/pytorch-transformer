import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from model import build_transformer, Transformer

import datasets
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilingualDataset

from tqdm import tqdm

from config import get_config

from train import get_ds
from torchviz import make_dot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], N=1, d_model=config['d_model']).to(device)


for batch in train_dataloader:
    model.eval()
    encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
    decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
    encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
    decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
    # embeds = model.dropout(model.src_embed(encoder_input) + model.src_pos(encoder_input))
    src = model.src_embed(encoder_input)
    src = model.src_pos(src)
    encoder = model.encoder
    dot = make_dot(encoder(src, encoder_mask), dict(model.named_parameters()))
    encoder_output = encoder(src, encoder_mask)
    # dot = make_dot(model.src_embed(encoder_input))
    dot.render("encoder.dot")

    decoder = model.decoder
    dot = make_dot(decoder(src, encoder_output, encoder_mask, decoder_mask), dict(decoder.named_parameters()))
    dot.render("decoder.dot")
    break