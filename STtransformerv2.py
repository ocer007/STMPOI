import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        # 独立的 Q, K, V 映射层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mtx, attn_mask=None, key_padding_mask=None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # [B, T, C]

        B, T, C = x.shape

        # 分别计算 Q, K, V
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在是 [B, H, T, D]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T, T]
        x_mtx = x_mtx.unsqueeze(1).repeat(1, 2, 1, 1)
        attn_scores = torch.mul(attn_scores, x_mtx)

        # 在这里加入扰动 逐元素相乘

        if attn_mask is not None:
            attn_scores += attn_mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, T]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # [B, H, T, D]
        context = context.transpose(1, 2).contiguous().reshape(B, T, C)

        out = self.out_proj(context)

        if not self.batch_first:
            out = out.transpose(0, 1)  # [T, B, C]

        return out, attn_weights


class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, batch_first=True):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, x_mtx, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self-attention block
        attn_output, _ = self.self_attn(src, x_mtx, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed-forward block
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class TransformerModel1(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel1, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.nlayers = nlayers
        self.transformer_encoder = nn.ModuleList([
            SimpleTransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
            for _ in range(nlayers)
        ])
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, x_mtx, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = src
        for layer in self.transformer_encoder:
            x = layer(x, x_mtx, src_mask=src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        return x, out_poi, out_time


class TransformerModel2(nn.Module):
    def __init__(self, num_poi, num_geo, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel2, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.nlayers = nlayers
        self.transformer_encoder = nn.ModuleList([
            SimpleTransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
            for _ in range(nlayers)
        ])
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_geo = nn.Linear(embed_size, num_geo)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, x_mtx, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = src
        for layer in self.transformer_encoder:
            x = layer(x, x_mtx, src_mask=src_mask)
        out_poi = self.decoder_poi(x)
        out_geo = self.decoder_geo(x)
        return x, out_poi, out_geo


class TransformerModel3(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel3, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.nlayers = nlayers
        self.transformer_encoder = nn.ModuleList([
            SimpleTransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
            for _ in range(nlayers)
        ])
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, x_mtx, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = src
        for layer in self.transformer_encoder:
            x = layer(x, x_mtx, src_mask=src_mask)
        out_poi = self.decoder_poi(x)
        out_cat = self.decoder_cat(x)
        return x, out_poi, out_cat
