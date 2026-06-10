import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# 定义降维模型
class MultiEmbReshape(nn.Module):
    def __init__(self, input_dim=768, output_dim=320):
        super(MultiEmbReshape, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


import torch
import torch.nn as nn


class POIMoE(nn.Module):
    def __init__(self, expert_num, input_dim, output_dim):
        super(POIMoE, self).__init__()
        self.expert_num = expert_num
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim)
            ) for _ in range(self.expert_num)
        ])

        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, self.expert_num),
            nn.Softmax(dim=-1)
        )

    def forward(self, poi_embeddings, poi_multi_embeddings):
        combined_embeddings = torch.cat((poi_embeddings, poi_multi_embeddings), dim=1)
        gate_output = self.gate(combined_embeddings)
        expert_output = torch.stack([expert(combined_embeddings) for expert in self.experts], dim=1)
        poi_fusion_embeddings = torch.sum(expert_output * gate_output.unsqueeze(-1), dim=1)
        return poi_fusion_embeddings


class POIMoEv2(nn.Module):
    def __init__(self, expert_num, input_dim, output_dim):
        super(POIMoEv2, self).__init__()
        self.expert_num = expert_num
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim),
            ) for _ in range(self.expert_num)
        ])

        self.gate_id = nn.Sequential(
            nn.Linear(self.input_dim, self.expert_num),
            nn.Softmax(dim=-1)
        )

        self.gate_multi = nn.Sequential(
            nn.Linear(self.input_dim, self.expert_num),
            nn.Softmax(dim=-1)
        )
        initializer_weight = [0.5, 0.5]
        self.weight = nn.Parameter(torch.tensor(initializer_weight), requires_grad=True)

    def forward(self, poi_embeddings, poi_multi_embeddings):
        combined_embeddings = torch.cat((poi_embeddings, poi_multi_embeddings), dim=1)
        gate_id_output = self.gate_id(combined_embeddings)
        gate_multi_output = self.gate_multi(combined_embeddings)
        expert_output = torch.stack([expert(combined_embeddings) for expert in self.experts], dim=1)
        poi_id_output = self.weight[0] * torch.sum(expert_output * gate_id_output.unsqueeze(-1), dim=1) + poi_embeddings
        poi_multi_output = self.weight[1] * torch.sum(expert_output * gate_multi_output.unsqueeze(-1),
                                                      dim=1) + poi_multi_embeddings
        weight = self.weight.detach().tolist()
        return poi_id_output, poi_multi_output, weight


class POIMoEv3(nn.Module):
    def __init__(self, expert_num, input_id_dim, input_multi_dim, output_dim):
        super(POIMoEv3, self).__init__()
        self.expert_num = expert_num
        self.input_id_dim = input_id_dim
        self.input_multi_dim = input_multi_dim
        self.output_dim = output_dim

        self.experts_id = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_id_dim, self.output_dim),
            ) for _ in range(self.expert_num)
        ])

        self.experts_multi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_multi_dim, self.output_dim),
            ) for _ in range(self.expert_num)
        ])

        self.gate = nn.Sequential(
            nn.Linear(self.input_id_dim + self.input_multi_dim, 2 * self.expert_num),
            nn.Softmax(dim=-1)
        )

        initializer_weight = [0.5]
        self.weight = nn.Parameter(torch.tensor(initializer_weight), requires_grad=True)

    def forward(self, poi_embeddings, poi_multi_embeddings):
        combined_embeddings = torch.cat((poi_embeddings, poi_multi_embeddings), dim=1)
        gate_output = self.gate(combined_embeddings)

        expert_id_output = torch.stack([expert(poi_embeddings) for expert in self.experts_id],
                                       dim=1)
        expert_multi_output = torch.stack([expert(poi_multi_embeddings) for expert in self.experts_multi],
                                          dim=1)
        expert_num = expert_id_output.size(1)

        gate_output_id = gate_output[:, :expert_num]
        gate_output_multi = gate_output[:, expert_num:]

        # 使用拆分后的gate_output分别计算poi_id_output和poi_multi_output
        poi_id_output = self.weight[0] * torch.sum(expert_id_output * gate_output_id.unsqueeze(-1),
                                                   dim=1) + poi_embeddings
        poi_multi_output = torch.sum(expert_multi_output * gate_output_multi.unsqueeze(-1),
                                     dim=1)

        fusion_embeddings = torch.cat((poi_id_output, poi_multi_output), dim=1)

        weight = self.weight.detach().tolist()
        return fusion_embeddings, weight


class POIMoEv4(nn.Module):
    def __init__(self, expert_num, input_id_dim, input_multi_dim, output_dim):
        super(POIMoEv4, self).__init__()
        self.expert_num = expert_num
        self.input_id_dim = input_id_dim
        self.input_multi_dim = input_multi_dim
        self.output_dim = output_dim

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_id_dim + self.input_multi_dim, self.output_dim),
            ) for _ in range(self.expert_num)
        ])

        self.gate_id = nn.Sequential(
            nn.Linear(self.input_id_dim, self.expert_num),
            nn.Softmax(dim=-1)
        )

        self.gate_multi = nn.Sequential(
            nn.Linear(self.input_multi_dim, self.expert_num),
            nn.Softmax(dim=-1)
        )

        initializer_weight = [0.5]
        self.weight = nn.Parameter(torch.tensor(initializer_weight), requires_grad=True)

    def forward(self, poi_embeddings, poi_multi_embeddings):
        combined_embeddings = torch.cat((poi_embeddings, poi_multi_embeddings), dim=1)
        gate_id_output = self.gate_id(poi_embeddings)
        gate_multi_output = self.gate_multi(poi_multi_embeddings)

        expert_output = torch.stack([expert(combined_embeddings) for expert in self.experts],
                                    dim=1)

        poi_id_output = self.weight[0] * torch.sum(expert_output * gate_id_output.unsqueeze(-1),
                                                   dim=1) + poi_embeddings
        poi_multi_output = torch.sum(expert_output * gate_multi_output.unsqueeze(-1),
                                     dim=1)

        fusion_embeddings = torch.cat((poi_id_output, poi_multi_output), dim=1)

        weight = self.weight.detach().tolist()
        return fusion_embeddings, weight


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class IntervalDistEncoder(nn.Module):
    def __init__(self, interval_scale=1, interval_vocab_size=30, hidden_size=32):
        super(IntervalDistEncoder, self).__init__()
        self.interval_scale = interval_scale
        self.interval_vocab_size = interval_vocab_size
        self.distance_embedding = nn.Embedding(interval_vocab_size, hidden_size)

    def forward(self, distance):
        interval_raw = torch.log2(distance.clamp(min=0) + 1.0)  # (batch_size, seq_len-1)
        mul_value =  interval_raw * self.interval_scale
        interval_index = torch.floor(mul_value).long()  # (batch_size, seq_len-1)
        interval_index = interval_index.clamp(min=0, max=self.interval_vocab_size - 1)
        interval_embedding = self.distance_embedding(interval_index)
        return interval_embedding

class IntervalTimeEncoder(nn.Module):
    def __init__(self, interval_scale=1, interval_vocab_size=30, hidden_size=32):
        super(IntervalTimeEncoder, self).__init__()
        self.interval_scale = interval_scale
        self.interval_vocab_size = interval_vocab_size
        self.distance_embedding = nn.Embedding(interval_vocab_size, hidden_size)

    def forward(self, distance):
        interval_raw = torch.log2(distance.clamp(min=0) + 1.0)  # (batch_size, seq_len-1)
        mul_value =  interval_raw * self.interval_scale
        interval_index = torch.floor(mul_value).long()  # (batch_size, seq_len-1)
        interval_index = interval_index.clamp(min=0, max=self.interval_vocab_size - 1)
        interval_embedding = self.distance_embedding(interval_index)
        return interval_embedding

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


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
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

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat


class TransformerModel1(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel1, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
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

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        return x, out_poi, out_time


class TransformerModel2(nn.Module):
    def __init__(self, num_poi, num_geo, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel2, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
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

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_geo = self.decoder_geo(x)
        return x, out_poi, out_geo


class TransformerModel3(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel3, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
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

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_cat = self.decoder_cat(x)
        return x, out_poi, out_cat


class STMoE(nn.Module):
    def __init__(self, num_poi, input1_dim, input2_dim, input3_dim):
        super(STMoE, self).__init__()
        self.num_poi = num_poi
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.input3_dim = input3_dim

        self.gate = nn.Sequential(
            nn.Linear(self.input1_dim + self.input2_dim + self.input3_dim, 3),
            nn.Softmax(dim=-1)
        )

        self.decoder_poi1 = nn.Linear(input1_dim, num_poi)
        self.decoder_poi2 = nn.Linear(input2_dim, num_poi)
        self.decoder_poi3 = nn.Linear(input3_dim, num_poi)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi1.bias.data.zero_()
        self.decoder_poi2.bias.data.zero_()
        self.decoder_poi3.bias.data.zero_()

        self.decoder_poi1.weight.data.uniform_(-initrange, initrange)
        self.decoder_poi2.weight.data.uniform_(-initrange, initrange)
        self.decoder_poi3.weight.data.uniform_(-initrange, initrange)

    def forward(self, output1, output2, output3):
        out_poi1 = self.decoder_poi1(output1)
        out_poi2 = self.decoder_poi2(output2)
        out_poi3 = self.decoder_poi3(output3)

        out_gate = self.gate(torch.cat((output1, output2,output3), dim=-1))

        out_gate_expanded = out_gate.unsqueeze(-1)
        out_pois = torch.stack((out_poi1, out_poi2, out_poi3), dim=2)
        out = (out_gate_expanded * out_pois).sum(dim=2)
        return out

class STMoEv2(nn.Module):
    def __init__(self, num_poi, input1_dim, input2_dim, input3_dim):
        super(STMoEv2, self).__init__()
        self.num_poi = num_poi
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.input3_dim = input3_dim

        self.gate = nn.Sequential(
            nn.Linear(self.input1_dim + self.input2_dim + self.input3_dim, 3),
            nn.Softmax(dim=-1)
        )

        self.decoder_poi1 = nn.Linear(input1_dim, num_poi)
        self.decoder_poi2 = nn.Linear(input2_dim, num_poi)
        self.decoder_poi3 = nn.Linear(input3_dim, num_poi)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi1.bias.data.zero_()
        self.decoder_poi2.bias.data.zero_()
        self.decoder_poi3.bias.data.zero_()

        self.decoder_poi1.weight.data.uniform_(-initrange, initrange)
        self.decoder_poi2.weight.data.uniform_(-initrange, initrange)
        self.decoder_poi3.weight.data.uniform_(-initrange, initrange)

    def forward(self, output1, output2, output3):
        out_poi1 = self.decoder_poi1(output1)
        out_poi2 = self.decoder_poi2(output2)
        out_poi3 = self.decoder_poi3(output3)

        out_gate = self.gate(torch.cat((output1, output2,output3), dim=-1))

        out_gate_expanded = out_gate.unsqueeze(-1)
        out_pois = torch.stack((out_poi1, out_poi2, out_poi3), dim=2)
        out = (out_gate_expanded * out_pois).sum(dim=2)
        return out_poi1,out_poi2,out_poi3,out


class TransformerModel_time(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_time, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
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

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        return out_poi, out_time


class TransformerModel_dist(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_dist, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_dist = nn.Linear(embed_size, 1)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_dist = self.decoder_dist(x)
        return out_poi, out_dist


class TransformerModel_multi(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel_multi, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
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

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_cat


class MultiTransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(MultiTransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
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

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_cat
