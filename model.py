from typing import Sequence
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LayerNormalization(nn.Module):

    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
        # print(f'd_model = {d_model}, d_ff = {d_ff}')
        # print(f'self.ll_1 param cnt = {count_parameters(self.linear_1)}')
        # print(f'self.dropout param cnt = {count_parameters(self.dropout)}')
        # print(f'self.ll_2 param cnt = {count_parameters(self.linear_2)}')

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# class InputEmbedding(nn.Module):

#     def __init__(self, d_model: int, vocab_size: int) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.vocab_size = vocab_size
#         self.embedding = nn.Embedding(vocab_size, d_model)

#     def forward(self, x):
#         # (batch, seq_len) --> (batch, seq_len, d_model)
#         # Multiply by sqrt(d_model) to scale the embeddings according to the paper
#         return self.embedding(x) * math.sqrt(self.d_model)

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size : int, model_d : int):
        super().__init__()
        self.model_d = model_d
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_d)

    def forward(self, xb : torch.Tensor):
        return self.embedding(xb) * (self.model_d ** 0.5)
    
# class PositionEmbedding(nn.Module):

#     def __init__(self, d_model: int, seq_len: int) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.seq_len = seq_len
#         # Create a matrix of shape (seq_len, d_model)
#         pe = torch.zeros(seq_len, d_model)
#         # Create a vector of shape (seq_len)
#         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
#         # Create a vector of shape (d_model)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
#         # Apply sine to even indices
#         pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
#         # Apply cosine to odd indices
#         pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
#         # Add a batch dimension to the positional encoding
#         pe = pe.unsqueeze(0) # (1, seq_len, d_model)
#         # Register the positional encoding as a buffer
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)


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
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # unqueeze to make batch dimension
        # RESOLVED: Is the unsqueeze unnecessary?
        # It's not!
        # self.pe = self.pe.unsqueeze(0).to(self.device) # (1, max_seq_len, model_d)
        self.register_buffer('pe', pe.to(self.device))

    def forward(self, xb):
        return self.pe[:xb.shape[1], :].requires_grad_(False)


# class Residual(nn.Module):
    
#         def __init__(self, dropout: float) -> None:
#             super().__init__()
#             self.dropout = nn.Dropout(dropout)
#             self.norm = LayerNormalization()
    
#         def forward(self, x, sublayer):
#             return x + self.dropout(sublayer(self.norm(x)))

class Residual(nn.Module):
    def __init__(self, dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNormalization()

    def forward(self, xb : torch.Tensor, sub_layer : nn.Module):
        return xb + self.dropout(sub_layer(self.ln(xb)))

# class MultiheadAttention(nn.Module):

#     def __init__(self, d_model: int, h: int, dropout: float) -> None:
#         super().__init__()
#         self.d_model = d_model # Embedding vector size
#         self.h = h # Number of heads
#         # Make sure d_model is divisible by h
#         assert d_model % h == 0, "d_model is not divisible by h"

#         self.d_k = d_model // h # Dimension of vector seen by each head
#         self.w_q = nn.Linear(d_model, d_model) # Wq
#         self.w_k = nn.Linear(d_model, d_model) # Wk
#         self.w_v = nn.Linear(d_model, d_model) # Wv
#         self.w_o = nn.Linear(d_model, d_model) # Wo
#         self.dropout = nn.Dropout(dropout)

#     @staticmethod
#     def attention(query, key, value, mask, dropout: nn.Dropout):
#         d_k = query.shape[-1]
#         # Just apply the formula from the paper
#         # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
#         attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
#         if mask is not None:
#             # Write a very low value (indicating -inf) to the positions where mask == 0
#             attention_scores.masked_fill_(mask == 0, -1e9)
#         attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
#         if dropout is not None:
#             attention_scores = dropout(attention_scores)
#         # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
#         # return attention scores which can be used for visualization
#         return (attention_scores @ value), attention_scores

#     def forward(self, q, k, v, mask):
#         query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
#         key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
#         value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

#         # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
#         query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
#         key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
#         value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

#         # Calculate attention
#         x, self.attention_scores = MultiheadAttention.attention(query, key, value, mask, self.dropout)
        
#         # Combine all the heads together
#         # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
#         x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

#         # Multiply by Wo
#         # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
#         return self.w_o(x)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model : int, head_cnt : int, dropout : float):
        super().__init__()
        self.head_cnt = head_cnt
        self.d_model = d_model
        assert d_model % head_cnt == 0
        self.d_k = d_model // head_cnt
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
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


# class EncoderBlock(nn.Module):

#     def __init__(self, self_attention_block: MultiheadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
#         super().__init__()
#         self.self_attention_block = self_attention_block
#         self.feed_forward_block = feed_forward_block
#         self.residual_connections = nn.ModuleList([Residual(dropout) for _ in range(2)])

#     def forward(self, x, src_mask):
#         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
#         x = self.residual_connections[1](x, self.feed_forward_block)
#         return x

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
    
# class DecoderBlock(nn.Module):

#     def __init__(self, self_attention_block: MultiheadAttention, cross_attention_block: MultiheadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
#         super().__init__()
#         self.self_attention_block = self_attention_block
#         self.cross_attention_block = cross_attention_block
#         self.feed_forward_block = feed_forward_block
#         self.residual_connections = nn.ModuleList([Residual(dropout) for _ in range(3)])

#     def forward(self, x, encoder_output, src_mask, tgt_mask):
#         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
#         x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
#         x = self.residual_connections[2](x, self.feed_forward_block)
#         return x

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

# class Encoder(nn.Module):

#     def __init__(self, layers: nn.ModuleList) -> None:
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNormalization()

#     def forward(self, x, mask):
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)

class Encoder(nn.Module):
    def __init__(self, encoder_blocks : Sequence[EncoderBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(encoder_blocks)
        self.ln = LayerNormalization()

    def forward(self, xb, mask):
        for block in self.blocks:
            xb = block(xb, mask)
        return self.ln(xb)
    
# class Decoder(nn.Module):

#     def __init__(self, layers: nn.ModuleList) -> None:
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNormalization()

#     def forward(self, x, encoder_output, src_mask, tgt_mask):
#         for layer in self.layers:
#             x = layer(x, encoder_output, src_mask, tgt_mask)
#         return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, decoder_blocks : Sequence[DecoderBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(decoder_blocks)
        self.ln = LayerNormalization()
    
    def forward(self, xb, encoder_output, src_mask, tgt_mask):
        for block in self.blocks:
            xb = block(xb, encoder_output, src_mask, tgt_mask)
        return self.ln(xb)

# class ProjectionLayer(nn.Module):

#     def __init__(self, d_model, vocab_size) -> None:
#         super().__init__()
#         self.proj = nn.Linear(d_model, vocab_size)

#     def forward(self, x) -> None:
#         # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
#         return torch.log_softmax(self.proj(x), dim = -1)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.ll = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.ll(x), dim=-1)
    
# class Transformer(nn.Module):

#     def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionEmbedding, tgt_pos: PositionEmbedding, projection_layer: ProjectionLayer, dropout=0.1) -> None:
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.src_pos = src_pos
#         self.tgt_pos = tgt_pos
#         self.projection_layer = projection_layer
#         self.dropout = nn.Dropout(dropout)

#     def encode(self, src, src_mask):
#         # (batch, seq_len, d_model)
#         src = self.dropout(self.src_embed(src) + self.src_pos(src))
#         return self.encoder(src, src_mask)
    
#     def decode(self, encoder_output: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
#         # (batch, seq_len, d_model)
#         tgt = self.dropout(self.tgt_embed(tgt) + self.tgt_pos(tgt))
#         return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
#     def project(self, x):
#         # (batch, seq_len, vocab_size)
#         return self.projection_layer(x)

class Transformer(nn.Module):
    def __init__(self, src_embed: InputEmbedding, tgt_embed : InputEmbedding, src_pos: PositionEmbedding, tgt_pos: PositionEmbedding, encoder : Encoder, decoder : Decoder, projection_layer : ProjectionLayer, dropout=0.1):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.dropout = nn.Dropout(dropout)

    def encode(self, xb, src_mask):
        print(f'type = {type(self.src_embed)}')
        self.src_embed(xb)
        self.src_pos(xb)
        embeds = self.dropout(self.src_embed(xb) + self.src_pos(xb))
        return self.encoder(embeds, src_mask)
    
    def decode(self, encoder_output, xb, src_mask, tgt_mask):
        embeds = self.dropout(self.tgt_embed(xb) + self.tgt_pos(xb))
        return self.decoder(embeds, encoder_output, src_mask, tgt_mask)
    
    def project(self, xb):
        return self.projection_layer(xb)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)

    # Create the positional encoding layers
    src_pos = PositionEmbedding(src_seq_len, d_model)
    tgt_pos = PositionEmbedding(tgt_seq_len, d_model)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        # print(f'self_attention_block param cnt = {count_parameters(encoder_self_attention_block)}')
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        # print(f'ffwd param cnt = {count_parameters(feed_forward_block)}')
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        # print(f'encoder_block param cnt = {count_parameters(encoder_block)}')
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        # print(f'decoder_block param cnt = {count_parameters(decoder_block)}')
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    # encoder = Encoder(nn.ModuleList(encoder_blocks))
    encoder = Encoder(encoder_blocks)
    # decoder = Decoder(nn.ModuleList(decoder_blocks))
    decoder = Decoder(decoder_blocks)
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    # print(f'projection_layer param cnt = {count_parameters(projection_layer)}')

    # Create the transformer
    transformer = Transformer(src_embed, tgt_embed, src_pos, tgt_pos, encoder, decoder, projection_layer, dropout=0.1) 
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # print(f'param cnt = {count_parameters(transformer)}')    
    return transformer

# def count_parameters(model):
#   return sum(p.numel() for p in model.parameters())