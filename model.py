# Reproduce loss: first 10 global steps
# loss = 10.017609596252441                                                                                                                                                                                                                                                                                                                
# loss = 9.848834991455078                                                                                                                                                                                                                                                                                                                 
# loss = 9.793350219726562                                                                                                                                                                                                                                                                                                                 
# loss = 9.768488883972168                                                                                                                                                                                                                                                                                                                 
# loss = 9.680366516113281                                                                                                                                                                                                                                                                                                                 
# loss = 9.612157821655273                                                                                                                                                                                                                                                                                                                 
# loss = 9.63258171081543                                                                                                                                                                                                                                                                                                                  
# loss = 9.568682670593262                                                                                                                                                                                                                                                                                                                 
# loss = 9.421436309814453                                                                                                                                                                                                                                                                                                                 
# loss = 9.477269172668457

from typing import Sequence, Union
import torch
import torch.nn.functional as F
from torch import nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size : int, model_d : int):
        super().__init__()
        self.model_d = model_d
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_d)

    def forward(self, xb : torch.Tensor):
        return self.embedding(xb) * (self.model_d ** 0.5)
    

class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len : int, d_model : int):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        pe = torch.zeros([max_seq_len, d_model])
        
        # div.shape = (d_model / 2)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        # pos.shape = (max_seq_len, 1)
        pos = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        # self.pe : (max_seq_len, d_model)
        # Here both pos and div got broadcasted.
        # ERROR 1: * 1/x and / x produce difference results
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # unqueeze to make batch dimension
        # RESOLVED: Is the unsqueeze unnecessary?
        # It's not!
        # self.pe = self.pe.unsqueeze(0).to(self.device) # (1, max_seq_len, model_d)
        self.register_buffer('pe', pe.to(self.device))

    def forward(self, xb):
        return self.pe[:xb.shape[1], :].requires_grad_(False)


class LayerNormalization(nn.Module):
    def __init__(self, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, xb):
        # xb: (B, T, H)
        # 这里xb可以是任何shape
        mean = xb.mean(dim = -1, keepdim = True) # (B, T, 1)
        std = xb.std(dim = -1, keepdim = True) # (B, T, 1)
        # 公式里的eps在根号下，这里略有不同
        return self.alpha * (xb - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model : int, d_ff : int, dropout : float):
        super().__init__()
        self.ll_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.ll_2 = nn.Linear(d_ff, d_model)
        # print(f'd_model = {d_model}, d_ff = {d_ff}')
        # print(f'self.ll_1 param cnt = {count_parameters(self.ll_1)}')
        # print(f'self.dropout param cnt = {count_parameters(self.dropout)}')
        # print(f'self.ll_2 param cnt = {count_parameters(self.ll_2)}')

    def forward(self, xb):
        xb = self.dropout(torch.relu(self.ll_1(xb)))
        xb = self.ll_2(xb)
        return xb

class MultiheadAttention(nn.Module):
    def __init__(self, d_model : int, head_cnt : int, dropout : float):
        super().__init__()
        self.head_cnt = head_cnt
        self.d_model = d_model
        assert d_model % head_cnt == 0
        self.d_k = d_model // head_cnt
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q_src, k_src, v_src : torch.Tensor, mask : torch.Tensor):
        def attention(q, k, v):
            d_k = q.shape[-1]
            logits = q @ k.transpose(-1, -2) / (d_k ** 0.5)
            # Sqrt and ** gives the same result
            # logits = (q @ k.transpose(-1, -2)) / math.sqrt(d_k)
            if mask is not None:
                # -1e9 and -float('inf') gave the the same result
                # logits.masked_fill_(mask == 0, -float('inf'))
                # ERROR 2
                # masked_fill_ is in place but masked_fill is not
                logits.masked_fill_(mask == 0, -1e9)
            probs = F.softmax(logits, dim = -1)
            # probs = logits.softmax(dim=-1)
            if self.dropout is not None:
                probs = self.dropout(probs)
            return (probs @ v), probs

        B = q_src.shape[0] # Batch size is always the same for different tensors.
        # all q,k,v have shape = B, T, d_model
        # T may be different among q, k, v.
        q = self.query(q_src)
        k = self.key(k_src)
        v = self.value(v_src)

        # q.view(...), k.view(...), v.view(...) have shape = B, T, hc, dk
        # after transpose it becomes B, hc, T, dk
        # all q, k, v should have the same batch size B.
        q = q.view(B, q.shape[1], self.head_cnt, self.d_k).transpose(1, 2)
        k = k.view(B, k.shape[1], self.head_cnt, self.d_k).transpose(1, 2)
        v = v.view(B, v.shape[1], self.head_cnt, self.d_k).transpose(1, 2)

        # B, hc, T, dk -> B, T, hc, dk -> B, T, d_model
        a, self.attention_scores = attention(q, k, v)
        a = a.transpose(1, 2).contiguous()
        a = a.view(B, a.shape[1], self.d_model) # transpose 和reshape连在一起，a.shape[1] evalutate的顺序不对。

        return self.output(a)
        

class Residual(nn.Module):
    def __init__(self, dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNormalization()

    def forward(self, xb : torch.Tensor, sub_layer : nn.Module):
        return xb + self.dropout(sub_layer(self.ln(xb)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention : MultiheadAttention, ffwd : FeedForward, dropout : float):
        super().__init__()
        self.self_attention = self_attention
        self.ffwd = ffwd
        self.residual_1 = Residual(dropout)
        self.residual_2 = Residual(dropout)

    def forward(self, xb, mask):
        xb = self.residual_1(xb, lambda x : self.self_attention(x, x, x, mask))
        xb = self.residual_2(xb, self.ffwd)
        return xb
        

class Encoder(nn.Module):
    def __init__(self, encoder_blocks : Sequence[EncoderBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(encoder_blocks)
        self.ln = LayerNormalization()

    def forward(self, xb, mask):
        for block in self.blocks:
            xb = block(xb, mask)
        return self.ln(xb)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention : MultiheadAttention, cross_attention : MultiheadAttention, ffwd : FeedForward, dropout : float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ffwd = ffwd
        self.residual_1 = Residual(dropout)
        self.residual_2 = Residual(dropout)
        self.residual_3 = Residual(dropout)

    
    def forward(self, xb, encoder_output, src_mask, tgt_mask):
        xb = self.residual_1(xb, lambda x : self.self_attention(x, x, x, tgt_mask))
        xb = self.residual_2(xb, lambda x : self.cross_attention(x, encoder_output, encoder_output, src_mask))
        xb = self.residual_3(xb, self.ffwd)
        return xb
    
class Decoder(nn.Module):
    def __init__(self, decoder_blocks : Sequence[DecoderBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(decoder_blocks)
        self.ln = LayerNormalization()
    
    def forward(self, xb, encoder_output, src_mask, tgt_mask):
        for block in self.blocks:
            xb = block(xb, encoder_output, src_mask, tgt_mask)
        return self.ln(xb)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.ll = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.ll(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, src_embed: InputEmbedding, tgt_embed : InputEmbedding, src_pos: PositionEmbedding, tgt_pos: PositionEmbedding, encoder : Encoder, decoder : Decoder, projection_layer : ProjectionLayer, dropout=0.1):
        super().__init__()
        # ERROR 3:
        # The order of class members is the order of initiation,
        # when we call nn.init.xavier_uniform_
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer
        self.dropout = nn.Dropout(dropout)

    def encode(self, xb, src_mask):
        embeds = self.dropout(self.src_embed(xb) + self.src_pos(xb))
        return self.encoder(embeds, src_mask)
    
    def decode(self, encoder_output, xb, src_mask, tgt_mask):
        embeds = self.dropout(self.tgt_embed(xb) + self.tgt_pos(xb))
        return self.decoder(embeds, encoder_output, src_mask, tgt_mask)
    
    def project(self, xb):
        return self.projection_layer(xb)
    

def build_transformer(src_vocab_size : int, tgt_vocab_size : int, src_max_seq_len: int, tgt_max_seq_len: int, d_model: 512, n_layer : int = 6, n_head: int = 8, dropout : float = 0.1, d_ff = 2048) -> Transformer:
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)

    src_pos = PositionEmbedding(src_max_seq_len, d_model)
    tgt_pos = PositionEmbedding(tgt_max_seq_len, d_model)

    encoder_blocks = []
    for _ in range(n_layer):
        self_attention_block = MultiheadAttention(d_model, n_head, dropout)
        # print(f'self_attention_block param cnt = {count_parameters(self_attention_block)}')
        ffwd = FeedForward(d_model, d_ff, dropout)
        # print(f'ffwd param cnt = {count_parameters(ffwd)}')
        encoder_block = EncoderBlock(self_attention_block, ffwd, dropout)
        # print(f'encoder_block param cnt = {count_parameters(encoder_block)}')
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(n_layer):
        self_attention_block = MultiheadAttention(d_model, n_head, dropout)
        cross_attention_block = MultiheadAttention(d_model, n_head, dropout)
        ffwd = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, ffwd, dropout)
        # print(f'decoder_block param cnt = {count_parameters(decoder_block)}')
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    # print(f'projection_layer param cnt = {count_parameters(projection_layer)}')

    transformer = Transformer(src_embed, tgt_embed, src_pos, tgt_pos, encoder, decoder, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    
    # print(f'param cnt = {count_parameters(transformer)}')
    return transformer

# def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
#     # Create the embedding layers
#     src_embed = InputEmbedding(d_model, src_vocab_size)
#     tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

#     # Create the positional encoding layers
#     src_pos = PositionEmbedding(d_model, src_seq_len)
#     tgt_pos = PositionEmbedding(d_model, tgt_seq_len)
    
#     # Create the encoder blocks
#     encoder_blocks = []
#     for _ in range(N):
#         encoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
#         print(f'self_attention_block param cnt = {count_parameters(encoder_self_attention_block)}')
#         feed_forward_block = FeedForward(d_model, d_ff, dropout)
#         print(f'ffwd param cnt = {count_parameters(feed_forward_block)}')
#         encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
#         print(f'encoder_block param cnt = {count_parameters(encoder_block)}')
#         encoder_blocks.append(encoder_block)

#     # Create the decoder blocks
#     decoder_blocks = []
#     for _ in range(N):
#         decoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
#         decoder_cross_attention_block = MultiheadAttention(d_model, h, dropout)
#         feed_forward_block = FeedForward(d_model, d_ff, dropout)
#         decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
#         print(f'decoder_block param cnt = {count_parameters(decoder_block)}')
#         decoder_blocks.append(decoder_block)
    
#     # Create the encoder and decoder
#     encoder = Encoder(nn.ModuleList(encoder_blocks))
#     decoder = Decoder(nn.ModuleList(decoder_blocks))
    
#     # Create the projection layer
#     projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
#     print(f'projection_layer param cnt = {count_parameters(projection_layer)}')

#     # Create the transformer
#     transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer) 
#     # Initialize the parameters
#     for p in transformer.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     print(f'param cnt = {count_parameters(transformer)}')    
#     return transformer

def count_parameters(model):
  return sum(p.numel() for p in model.parameters())

# d_model = 10
# # xb : (2, 3)
# xb = torch.arange(0,6).reshape([2,3])
# # print(f'xb.shape = {xb.shape}')

# # tok_embedding : (2, 3, 10)
# tok_embedding = InputEmbedding(20, 10)(xb)
# print(f'tok_embedding.shape = {tok_embedding.shape}')

# # pos_embedding : (1, 3, 10)
# pos_embedding = PositionEmbedding(20, 10)(xb)
# print(f'pos_embedding.shape = {pos_embedding.shape}')

# # embedding : (2, 3, 10)
# embedding = tok_embedding + pos_embedding
# print(f'embedding.shape = {embedding.shape}')

# # TODO
# # Dropout(tok_embedding + pos_embedding)

# # embedding : (2, 3, 10)
# embedding = LayerNormalization()(embedding)
# print(f'embedding.shape = {embedding.shape}')

# # embedding : (2, 3, 10)
# embedding = MultiheadAttention(d_model, 5)(embedding, embedding, embedding)
# print(f'embedding.shape = {embedding.shape}')

# # embedding : (2, 3, 10)
# embedding = FeedForward(d_model, d_model * 4, dropout = 0.2)(embedding)
# print(f'embedding.shape = {embedding.shape}')

# # embedding : (2, 3, 10)
# embedding = Residual(dropout = 0.2)(embedding, lambda x : MultiheadAttention(d_model, 5)(embedding, embedding, embedding))
# print(f'embedding.shape = {embedding.shape}')

# encoderBlock = EncoderBlock(MultiheadAttention(d_model, 5), FeedForward(d_model, d_model * 4, dropout=0.2), 0.2)
# embedding = encoderBlock(embedding, None)
# print(f'embedding.shape = {embedding.shape}')

# encoder_blocks = []
# for _ in range(3):
#     encoder_blocks.append(EncoderBlock(MultiheadAttention(d_model, 5), FeedForward(d_model, d_model * 4, dropout=0.2), 0.2))
# encoder = Encoder(encoder_blocks)
# encoder_output = encoder(embedding, None)
# print(f'ecnoder_output.shape = {encoder_output.shape}')










# ab = torch.arange(0,8).reshape([2,4])
# print(f'ab.shape = {ab.shape}')
# ab_tok_embedding = InputEmbedding(20, 10)(ab)
# ab_pos_embedding = PositionEmbedding(20, 10)(ab)
# ab = ab_tok_embedding + ab_pos_embedding
# print(f'ab.shape = {ab.shape}')

# decoder_blocks = []
# for _ in range(3):
#     decoder_blocks.append(DecoderBlock(MultiheadAttention(d_model, 5), MultiheadAttention(d_model, 5), FeedForward(d_model, d_model * 4, dropout=0.2), 0.2))
# decoder = Decoder(decoder_blocks)
# decoder_output = decoder(ab, encoder_output, None, None)
# print(f'decoder_output.shape = {decoder_output.shape}')

# probs = ProjectionLayer(d_model, 30)(decoder_output)
# print(f'probs.shape = {probs.shape}')



# build_transformer(10, 20, 20, 40, 30, 6, 8, 0.1, 120)