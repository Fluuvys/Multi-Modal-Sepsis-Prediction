"""
utde.py -- UTDE / MulT-cross irregular-time multimodal baseline, adapted into
experiments/train.py's MODEL_REGISTRY.

SOURCE: Zhang, Li, Chen, Yan, Petzold. "Improving Medical Predictions by Irregular
Multimodal Electronic Health Records Modeling." ICML 2023. arXiv:2210.12156.
Code: github.com/XZhang97666/MultimodalMIMIC (model.py / module.py / interp.py, as of
the version pasted into this project's chat history -- no LICENSE file was present in
that repo at the time this was vendored; keep this attribution if you redistribute
anything beyond internal baseline reproduction, and confirm terms with the authors
before any wider release, per PROJECT_CONTEXT.md rule #1's "independently reproduced"
requirement).

Everything from here down to "ADAPTER" is the vendored model code, kept as close to the
original as possible (only the `transformers` import list was trimmed -- AdamW,
get_scheduler, set_seed, BertPreTrainedModel, AutoConfig, BertTokenizer were imported in
the original but never referenced, and several aren't top-level exports in current
transformers versions anyway).

KNOWN ISSUE IN THE VENDORED CODE (not introduced by this adapter): BertForRepresentation
stacks per-note embeddings sequence-first (`torch.stack(txt_arr)` -> [n_notes, B,
hidden]), but multiTimeAttention.forward immediately unpacks `value.size()` as
`(batch, seq_len, dim)` -- batch-first. Fed straight through on the text/TS_Text path,
`n_notes` silently becomes the batch dimension. UTDEBaseline below only wires up the
TS-only arm (`modeltype="TS"`), which never touches this code path, so it's unaffected --
but don't trust a TS_Text run built on top of this file until you've traced that.
"""
from __future__ import annotations

import math
import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter, BCELoss, CrossEntropyLoss
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertModel

# ==========================================
# 1. BASE MODULES
# ==========================================

class Outer(nn.Module):
    def __init__(self, inp1_size: int = 128, inp2_size: int = 128, n_neurons: int = 128):
        super(Outer, self).__init__()
        self.inp1_size = inp1_size
        self.inp2_size = inp2_size
        self.feedforward = nn.Sequential(
            nn.Linear((inp1_size + 1) * (inp2_size + 1), n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
        )

    def forward(self, inp1, inp2):
        batch_size = inp1.size(0)
        append = torch.ones((batch_size, 1)).type_as(inp1)
        inp1 = torch.cat([inp1, append], dim=-1)
        inp2 = torch.cat([inp2, append], dim=-1)
        fusion = torch.zeros((batch_size, self.inp1_size + 1, self.inp2_size + 1)).type_as(inp1)
        for i in range(batch_size):
            fusion[i] = torch.outer(inp1[i], inp2[i])
        fusion = fusion.flatten(1)
        return self.feedforward(fusion)

class MAGGate(nn.Module):
    def __init__(self, inp1_size, inp2_size, dropout):
        super(MAGGate, self).__init__()
        self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc3 = nn.Linear(inp2_size, inp1_size)
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2):
        w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
        adjust = self.fc3(w2 * inp2)
        one = torch.tensor(1).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust
        output = self.dropout(self.norm(output))
        return output

class gateMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid()
        )
        self._initialize()

    def _initialize(self):
        for model in [self.gate]:
            for layer in model:
                if type(layer) in [nn.Linear]:
                    torch.nn.init.xavier_normal_(layer.weight)

    def forward(self, hidden_states):
        gate_logits = self.gate(hidden_states)
        return gate_logits

class TimeSeriesCnnModel(nn.Module):
    def __init__(self, input_size, n_filters, filter_size, dropout, length, n_neurons, layers):
        super().__init__()
        padding = int(np.floor(filter_size / 2))
        self.layers = layers
        if layers >= 1:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
        if layers >= 2:
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
        if layers >= 3:
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(int(length * n_filters / (2**layers)), n_neurons)
        self.fc1_drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.layers >= 1: x = self.pool1(F.relu(self.conv1(x)))
        if self.layers >= 2: x = self.pool2(F.relu(self.conv2(x)))
        if self.layers >= 3: x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        return x

class multiTimeAttention(nn.Module):
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape) == 3: mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=0.1):
        batch, seq_len, dim = value.size()
        if mask is not None: mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2) for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        tgt_len, bsz, embed_dim = query.size()

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            if key is None:
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None: bias = bias[start:end]
        return F.linear(input, weight, bias)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx, init_size=1024, auto_expand=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.register_buffer(
            "weights",
            SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim, padding_idx),
            persistent=False,
        )
        self.max_positions = int(1e5)
        self.auto_expand = auto_expand

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, positions=None):
        bsz, seq_len = input.size(0), input.size(1)
        max_pos = self.padding_idx + 1 + seq_len
        weights = self.weights

        if max_pos > self.weights.size(0):
            weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            ).to(self.weights)
            if self.auto_expand:
                self.weights = weights

        if incremental_state is not None:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        mask = input.ne(self.padding_idx).long()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def LayerNorm(embedding_dim):
    return nn.LayerNorm(embedding_dim)

def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)

def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None: dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda: future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, attn_dropout=attn_dropout)
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, device, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, embed_dropout=0.0, attn_mask=False, learn_embed=True, q_seq_len=None, kv_seq_len=None):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device = device
        self.q_seq_len = q_seq_len
        self.kv_seq_len = kv_seq_len
        if learn_embed:
            if self.q_seq_len is not None:
                self.embed_positions_q = nn.Embedding(self.q_seq_len, embed_dim, padding_idx=0)
                nn.init.normal_(self.embed_positions_q.weight, std=0.02)
            if self.kv_seq_len is not None:
                self.embed_positions_kv = nn.Embedding(self.kv_seq_len, embed_dim)
                nn.init.normal_(self.embed_positions_kv.weight, std=0.02)
        else:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, padding_idx=0)

        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads=num_heads, attn_dropout=attn_dropout, relu_dropout=relu_dropout, res_dropout=res_dropout, attn_mask=attn_mask) for _ in range(layers)])
        self.normalize = True
        if self.normalize: self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        x = x_in
        length_x = x.size(0)
        x = self.embed_scale * x_in
        if self.q_seq_len is not None:
            position_x = torch.arange(length_x, dtype=torch.long, device=self.device)
            x += (self.embed_positions_q(position_x).unsqueeze(0)).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            length_kv = x_in_k.size(0)
            position_kv = torch.arange(length_kv, dtype=torch.long, device=self.device)
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.kv_seq_len is not None:
                x_k += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0, 1)
                x_v += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)

        if self.normalize: x = self.layer_norm(x)
        return x

class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pre_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])
        self.self_attns = nn.ModuleList([MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, attn_dropout=attn_dropout) for _ in range(2)])
        self.pre_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])
        self.cross_attn_1 = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, attn_dropout=attn_dropout)
        self.cross_attn_2 = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, attn_dropout=attn_dropout)
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.fc1 = nn.ModuleList([nn.Linear(self.embed_dim, 4*self.embed_dim) for _ in range(2)])
        self.fc2 = nn.ModuleList([nn.Linear(4*self.embed_dim, self.embed_dim) for _ in range(2)])
        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x_list):
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_self_attn_layer_norm, x_list)]
        output = [l(query=x, key=x, value=x) for l, x in zip(self.self_attns, x_list)]
        x_list = [x for x, _ in output]
        x_list[0] = F.dropout(x_list[0], p=self.res_dropout, training=self.training)
        x_list[1] = F.dropout(x_list[1], p=self.res_dropout, training=self.training)
        x_list = [r + x for r, x in zip(residual, x_list)]

        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_list)]
        x_txt, x_ts = x_list
        x_ts_to_txt, _ = self.cross_attn_1(query=x_txt, key=x_ts, value=x_ts)
        x_txt_to_ts, _ = self.cross_attn_2(query=x_ts, key=x_txt, value=x_txt)
        x_ts_to_txt = F.dropout(x_ts_to_txt, p=self.res_dropout, training=self.training)
        x_txt_to_ts = F.dropout(x_txt_to_ts, p=self.res_dropout, training=self.training)
        x_list = [r + x for r, x in zip(residual, (x_ts_to_txt, x_txt_to_ts))]

        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_ffn_layer_norm, x_list)]
        x_list = [F.relu(l(x)) for l, x in zip(self.fc1, x_list)]
        x_list[0] = F.dropout(x_list[0], p=self.relu_dropout, training=self.training)
        x_list[1] = F.dropout(x_list[1], p=self.relu_dropout, training=self.training)
        x_list = [l(x) for l, x in zip(self.fc2, x_list)]
        x_list[0] = F.dropout(x_list[0], p=self.res_dropout, training=self.training)
        x_list[1] = F.dropout(x_list[1], p=self.res_dropout, training=self.training)
        x_list = [r + x for r, x in zip(residual, x_list)]
        return x_list

class TransformerCrossEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, device, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, embed_dropout=0.0, attn_mask=False, q_seq_len_1=None, q_seq_len_2=None):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device = device
        self.q_seq_len_1 = q_seq_len_1
        self.q_seq_len_2 = q_seq_len_2
        self.embed_positions_q_1 = nn.Embedding(self.q_seq_len_1, embed_dim, padding_idx=0)
        nn.init.normal_(self.embed_positions_q_1.weight, std=0.02)
        if self.q_seq_len_2 is not None:
            self.embed_positions_q_2 = nn.Embedding(self.q_seq_len_2, embed_dim, padding_idx=0)
            nn.init.normal_(self.embed_positions_q_2.weight, std=0.02)
            self.embed_positions_q = nn.ModuleList([self.embed_positions_q_1, self.embed_positions_q_2])
        else:
            self.embed_positions_q = nn.ModuleList([self.embed_positions_q_1, self.embed_positions_q_1])

        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([TransformerCrossEncoderLayer(embed_dim, num_heads=num_heads, attn_dropout=attn_dropout, relu_dropout=relu_dropout, res_dropout=res_dropout, attn_mask=attn_mask) for _ in range(layers)])
        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

    def forward(self, x_in_list):
        x_list = x_in_list
        length_x1 = x_list[0].size(0)
        length_x2 = x_list[1].size(0)
        x_list = [self.embed_scale * x_in for x_in in x_in_list]
        if self.q_seq_len_1 is not None:
            position_x1 = torch.tensor(torch.arange(length_x1), dtype=torch.long).to(self.device)
            position_x2 = torch.tensor(torch.arange(length_x2), dtype=torch.long).to(self.device)
            positions = [position_x1, position_x2]
            x_list = [l(position_x).unsqueeze(0).transpose(0, 1) + x for l, x, position_x in zip(self.embed_positions_q, x_list, positions)]
        x_list[0] = F.dropout(x_list[0], p=self.dropout, training=self.training)
        x_list[1] = F.dropout(x_list[1], p=self.dropout, training=self.training)
        for layer in self.layers:
            x_list = layer(x_list)
        if self.normalize:
            x_list = [l(x) for l, x in zip(self.layer_norm, x_list)]
        return x_list

# ==========================================
# 2. INTERPOLATION MODULES
# ==========================================

def hold_out(mask, perc=0.2):
    mask = mask.cpu().detach().numpy()
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(0.20*count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(0.20*count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask

def recon_loss(x_ts, m1, m2, ypred, num_features):
    y = x_ts.transpose(1, 2)
    m1 = m1.transpose(1, 2)
    m2 = m2.transpose(1, 2)
    m2 = 1 - m2
    m = m1 * m2
    ypred = ypred[:, :num_features, :]
    x = (y - ypred) * (y - ypred)
    x = x * m
    count = torch.sum(m, dim=2)
    count = torch.where(count > 0, count, torch.ones_like(count))
    x = torch.sum(x, dim=2) / count
    x = torch.sum(x, dim=1) / num_features
    return torch.mean(x)

class S_Interp(nn.Module):
    def __init__(self, args, device, orig_d_ts):
        super(S_Interp, self).__init__()
        self.tt_max = args.tt_max
        self.device = device
        self.ref_t = torch.linspace(0, 1., self.tt_max).to(self.device)
        self.d_dim = orig_d_ts
        self.output = nn.Linear(args.embed_dim, args.embed_dim)
        self.kernel = Parameter(torch.zeros(self.d_dim))

    def forward(self, x_ts, x_ts_mask, ts_tt_list, rec_mask, reconstruction=False):
        x_ts = x_ts.transpose(1, 2)
        x_ts_mask = x_ts_mask.transpose(1, 2)
        tt_len = ts_tt_list.shape[-1]
        d = ts_tt_list.unsqueeze(1).repeat(1, self.d_dim, 1)
        if reconstruction:
            output_dim = tt_len
            m = rec_mask.transpose(1, 2)
            ref_t = d.unsqueeze(-2).repeat(1, 1, output_dim, 1)
        else:
            m = x_ts_mask
            ref_t = self.ref_t.unsqueeze(0)
            output_dim = self.tt_max

        d = d.unsqueeze(-1).repeat(1, 1, 1, output_dim)
        mask = m.unsqueeze(-1).repeat(1, 1, 1, output_dim)
        x_ts = x_ts.unsqueeze(-1).repeat(1, 1, 1, output_dim)
        norm = (d - ref_t)*(d - ref_t)
        a = torch.ones([self.d_dim, tt_len, output_dim]).to(self.device)

        pos_kernel = torch.log(1 + torch.exp(self.kernel))
        alpha = a * pos_kernel.unsqueeze(-1).unsqueeze(-1)
        w = torch.logsumexp(-alpha*norm + torch.log(mask+1e-12), dim=2)
        w1 = w.unsqueeze(2).repeat(1, 1, tt_len, 1)
        w1 = torch.exp(-alpha*norm + torch.log(mask+1e-12) - w1)
        y = torch.sum(w1*x_ts, dim=2)

        w_t = torch.logsumexp(-10.0*alpha*norm + torch.log(mask+1e-12), dim=2)
        w_t = w.unsqueeze(2).repeat(1, 1, tt_len, 1)
        w_t = torch.exp(-10.0*alpha*norm + torch.log(mask+1e-12) - w_t)
        y_trans = torch.sum(w_t*x_ts, dim=2)
        rep1 = torch.cat([y, w, y_trans], dim=1)
        return rep1

class Cross_Interp(nn.Module):
    def __init__(self, args, device, orig_d_ts):
        super(Cross_Interp, self).__init__()
        self.device = device
        self.d_dim = orig_d_ts
        self.activation = nn.Sigmoid()
        self.cross_channel_interp = torch.empty(self.d_dim, self.d_dim).to(self.device)
        nn.init.eye_(self.cross_channel_interp)

    def forward(self, x, reconstruction=False):
        self.output_dim = x.shape[-1]
        cross_channel_interp = self.cross_channel_interp
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim:2*self.d_dim, :]
        intensity = torch.exp(w)
        y = y.permute(0, 2, 1)
        w = w.permute(0, 2, 1)
        w2 = w
        w = w.unsqueeze(-1).repeat(1, 1, 1, self.d_dim)
        den = torch.logsumexp(w, dim=2)
        w = torch.exp(w2 - den)
        mean = torch.mean(y, dim=1).unsqueeze(1).repeat(1, self.output_dim, 1)
        w2 = torch.matmul(w*(y - mean), cross_channel_interp) + mean
        rep1 = w2.permute(0, 2, 1)
        if reconstruction is False:
            y_trans = x[:, 2*self.d_dim:3*self.d_dim, :]
            y_trans = y_trans - rep1
            rep1 = torch.cat([rep1, intensity, y_trans], 1)
        return rep1

# ==========================================
# 3. MAIN MODELS
# ==========================================

class BertForRepresentation(nn.Module):
    def __init__(self, args, BioBert):
        super().__init__()
        self.bert = BioBert
        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name = args.model_name

    def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None, doc_idx_list=None):
        txt_arr = []
        for input_ids, attention_mask in zip(input_ids_sequence, attention_mask_sequence):
            if 'Longformer' in self.model_name:
                attention_mask -= 1
                text_embeddings = self.bert(input_ids, global_attention_mask=attention_mask)
            else:
                text_embeddings = self.bert(input_ids, attention_mask=attention_mask)
            text_embeddings = text_embeddings[0][:,0,:]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)
        txt_arr = torch.stack(txt_arr)
        return txt_arr

class TextModel(nn.Module):
    def __init__(self, args, device, orig_d_txt=768, Biobert=None):
        super(TextModel, self).__init__()
        self.device = device
        self.task = args.task
        self.agg_type = args.agg_type
        self.out_dropout = args.dropout
        self.orig_d_txt = orig_d_txt
        self.d_txt = args.embed_dim
        self.bertrep = BertForRepresentation(args, Biobert)
        self.proj_txt = nn.Linear(self.orig_d_txt, self.d_txt)
        output_dim = args.num_labels

        self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        self.out_layer = nn.Linear(self.d_txt, output_dim)

        if self.task == 'ihm':
            self.loss_fct1 = CrossEntropyLoss()
        elif self.task == 'pheno':
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def forward(self, input_ids_sequences, attn_mask_sequences, labels=None):
        x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)
        x_txt = torch.mean(x_txt, dim=1)
        proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(proj_x_txt)), p=self.out_dropout, training=self.training))
        last_hs_proj += proj_x_txt
        output = self.out_layer(last_hs_proj)

        if self.task == 'ihm':
            if labels is not None: return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output, dim=-1)[:,1]
        elif self.task == 'pheno':
            if labels is not None: return self.loss_fct1(output, labels.float())
            return torch.nn.functional.sigmoid(output)

class MULTCrossModel(nn.Module):
    def __init__(self, args, device, modeltype=None, orig_d_ts=None, orig_reg_d_ts=None, orig_d_txt=None, ts_seq_num=None, text_seq_num=None, Biobert=None):
        super(MULTCrossModel, self).__init__()
        self.modeltype = modeltype if modeltype is not None else args.modeltype
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.device = device
        self.kernel_size = args.kernel_size
        self.dropout = args.dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts = args.irregular_learn_emb_ts
        self.irregular_learn_emb_text = args.irregular_learn_emb_text
        self.reg_ts = args.reg_ts
        self.TS_mixup = args.TS_mixup
        self.mixup_level = args.mixup_level
        self.task = args.task
        self.tt_max = args.tt_max
        self.cross_method = args.cross_method

        if self.irregular_learn_emb_ts or self.irregular_learn_emb_text:
            self.time_query = torch.linspace(0, 1., self.tt_max)
            self.periodic = nn.Linear(1, args.embed_time-1)
            self.linear = nn.Linear(1, 1)

        if "TS" in self.modeltype:
            self.orig_d_ts = orig_d_ts
            self.d_ts = args.embed_dim
            self.ts_seq_num = ts_seq_num
            if self.irregular_learn_emb_ts:
                self.time_attn_ts = multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)
            if self.reg_ts:
                self.orig_reg_d_ts = orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)
            if self.TS_mixup:
                if self.mixup_level in ['batch', 'batch_seq']:
                    self.moe = gateMLP(input_dim=self.d_ts*2, hidden_size=args.embed_dim, output_dim=1, dropout=args.dropout)
                elif self.mixup_level == 'batch_seq_feature':
                    self.moe = gateMLP(input_dim=self.d_ts*2, hidden_size=args.embed_dim, output_dim=self.d_ts, dropout=args.dropout)
                else:
                    raise ValueError("Unknown mixedup type")

        if "Text" in self.modeltype:
            self.orig_d_txt = orig_d_txt
            self.d_txt = args.embed_dim
            self.text_seq_num = text_seq_num
            self.bertrep = BertForRepresentation(args, Biobert)
            if self.irregular_learn_emb_text:
                self.time_attn = multiTimeAttention(768, self.d_txt, args.embed_time, 8)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        output_dim = args.num_labels
        if self.modeltype == "TS_Text":
            if self.cross_method == "self_cross":
                self.trans_self_cross_ts_txt = self.get_cross_network(layers=args.cross_layers)
            else:
                self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
                self.trans_txt_mem = self.get_network(self_type='txt_mem', layers=args.layers)
                if self.cross_method == "MulT":
                    self.trans_txt_with_ts = self.get_network(self_type='txt_with_ts', layers=args.cross_layers)
                    self.trans_ts_with_txt = self.get_network(self_type='ts_with_txt', layers=args.cross_layers)
                elif self.cross_method == "MAGGate":
                    self.gate_fusion = MAGGate(inp1_size=self.d_txt, inp2_size=self.d_ts, dropout=self.dropout)
                elif self.cross_method == "Outer":
                    self.outer_fusion = Outer(inp1_size=self.d_txt, inp2_size=self.d_ts)

            dim_mult = (self.d_ts + self.d_txt) if self.cross_method in ["self_cross", "MulT", "None"] else self.d_txt
            self.proj1 = nn.Linear(dim_mult, dim_mult)
            self.proj2 = nn.Linear(dim_mult, dim_mult)
            self.out_layer = nn.Linear(dim_mult, output_dim)

        if self.task == 'ihm': self.loss_fct1 = nn.CrossEntropyLoss()
        elif self.task == 'pheno': self.loss_fct1 = nn.BCEWithLogitsLoss()
        else: raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        if self_type == 'ts_mem':
            embed_dim, q_seq_len, kv_seq_len = self.d_ts, (self.tt_max if self.irregular_learn_emb_ts else self.ts_seq_num), None
        elif self_type == 'txt_mem':
            embed_dim, q_seq_len, kv_seq_len = self.d_txt, (self.tt_max if self.irregular_learn_emb_text else self.text_seq_num), None
        elif self_type == 'txt_with_ts':
            embed_dim, q_seq_len, kv_seq_len = self.d_ts, (self.tt_max if self.irregular_learn_emb_ts else self.text_seq_num), (self.tt_max if self.irregular_learn_emb_ts else self.ts_seq_num)
        elif self_type == 'ts_with_txt':
            embed_dim, q_seq_len, kv_seq_len = self.d_txt, (self.tt_max if self.irregular_learn_emb_text else self.ts_seq_num), (self.tt_max if self.irregular_learn_emb_text else self.text_seq_num)
        else: raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim, num_heads=self.num_heads, layers=layers, device=self.device, attn_dropout=self.dropout, relu_dropout=self.dropout, res_dropout=self.dropout, embed_dropout=self.dropout, attn_mask=self.attn_mask, q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)

    def get_cross_network(self, layers=-1):
        return TransformerCrossEncoder(embed_dim=self.d_ts, num_heads=self.num_heads, layers=layers, device=self.device, attn_dropout=self.dropout, relu_dropout=self.dropout, res_dropout=self.dropout, embed_dropout=self.dropout, attn_mask=self.attn_mask, q_seq_len_1=self.tt_max)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device).unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x_ts, x_ts_mask, ts_tt_list, input_ids_sequences, attn_mask_sequences, note_time_list, note_time_mask_list, labels=None, reg_ts=None):
        if "TS" in self.modeltype:
            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
                x_ts_mask_cat = torch.cat((x_ts_mask, x_ts_mask), 2)
                proj_x_ts_irg = self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask_cat).transpose(0, 1)

            if self.reg_ts and reg_ts is not None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.mixup_level == 'batch':
                    g_irg = torch.max(proj_x_ts_irg, dim=0).values
                    g_reg = torch.max(proj_x_ts_reg, dim=0).values
                    moe_gate = torch.cat([g_irg, g_reg], dim=-1)
                elif self.mixup_level in ['batch_seq', 'batch_seq_feature']:
                    moe_gate = torch.cat([proj_x_ts_irg, proj_x_ts_reg], dim=-1)
                mixup_rate = self.moe(moe_gate)
                proj_x_ts = mixup_rate * proj_x_ts_irg + (1 - mixup_rate) * proj_x_ts_reg
            else:
                proj_x_ts = proj_x_ts_irg if self.irregular_learn_emb_ts else proj_x_ts_reg

        if "Text" in self.modeltype:
            x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)
            if self.irregular_learn_emb_text:
                time_key = self.learn_time_embedding(note_time_list).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                proj_x_txt = self.time_attn(time_query, time_key, x_txt, note_time_mask_list).transpose(0, 1)
            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)

        if self.cross_method == "self_cross":
            hiddens = self.trans_self_cross_ts_txt([proj_x_txt, proj_x_ts])
            h_txt_with_ts, h_ts_with_txt = hiddens
            last_hs = torch.cat([h_txt_with_ts[-1], h_ts_with_txt[-1]], dim=1)
        else:
            if self.cross_method == "MulT":
                h_txt_with_ts = self.trans_txt_with_ts(proj_x_txt, proj_x_ts, proj_x_ts)
                h_ts_with_txt = self.trans_ts_with_txt(proj_x_ts, proj_x_txt, proj_x_txt)
                proj_x_ts = self.trans_ts_mem(h_txt_with_ts)
                proj_x_txt = self.trans_txt_mem(h_ts_with_txt)
                last_hs = torch.cat([proj_x_ts[-1], proj_x_txt[-1]], dim=1)
            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
                proj_x_txt = self.trans_txt_mem(proj_x_txt)
                if self.cross_method == "MAGGate":
                    last_hs = self.gate_fusion(proj_x_txt[-1], proj_x_ts[-1])
                elif self.cross_method == "Outer":
                    last_hs = self.outer_fusion(proj_x_txt[-1], proj_x_ts[-1])
                else:
                    last_hs = torch.cat([proj_x_txt[-1], proj_x_ts[-1]], dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        if self.task == 'ihm':
            if labels is not None: return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output, dim=-1)[:, 1]
        elif self.task == 'pheno':
            if labels is not None: return self.loss_fct1(output, labels.float())
            return torch.nn.functional.sigmoid(output)


# ==========================================
# 4. ADAPTER -- wires the vendored model into experiments/train.py's MODEL_REGISTRY
# ==========================================
# dataset.py's collate_sepsis_batch produces a dict, not the positional args
# MULTCrossModel.forward expects, and MULTCrossModel returns a LOSS when you pass
# labels= (its own CrossEntropyLoss/BCEWithLogitsLoss) rather than logits. Both need
# adapting so every baseline is scored identically through evaluate.py
# (PROJECT_CONTEXT.md rule #1) -- see this project's earlier adapter (mult_cross.py) for
# the full explanation; this is the same logic renamed for the utde.py entry point.

from dataset import VARIABLE_VOCAB  # adjust to `from experiments.dataset import ...`
# if you're not relying on experiments/ being on sys.path -- see the import-fragility
# note from earlier in this project's history

DEFAULT_MODEL_ARGS = dict(
    modeltype="TS_Text",          # now TS+notes by default -- see module docstring
    num_heads=8,
    layers=2,
    cross_layers=2,
    kernel_size=3,
    dropout=0.1,
    irregular_learn_emb_ts=True,
    irregular_learn_emb_text=True,
    reg_ts=False,
    TS_mixup=False,
    mixup_level="batch",
    task="pheno",                 # -> BCEWithLogitsLoss internally; we discard its output,
                                   # evaluate.py owns the loss for every model uniformly
    tt_max=48,                    # size of the fixed reference time-grid irregular
                                   # observations get re-aligned onto (shared by TS and text)
    cross_method="None",          # "None" = TS and text each pass through their own
                                   # self-attention memory (trans_ts_mem/trans_txt_mem),
                                   # fused only by concatenation before our head -- no
                                   # cross-modal attention. Set to "MulT" for real
                                   # cross-attention fusion once you want to compare.
    embed_dim=64,
    embed_time=16,
    num_labels=1,
    lookback_hours=48.0,          # TS time-grid normalization horizon, anchored to t
    text_lookback_hours=48.0,     # same idea, separate knob, for notes
    text_model_name="emilyalsentzer/Bio_ClinicalBERT",
    max_note_tokens=256,          # per-note truncation length passed to the tokenizer
    freeze_text_encoder=True,     # BERT weights frozen by default -- see rationale in
                                  # the class docstring. Set False for full fine-tuning.
    max_ts_events=500,            # hard cap on TS events fed into attention per sample,
                                  # AFTER lookback windowing -- see _window_events()
    max_notes=50,                 # same idea for notes, though counts are naturally
                                  # much smaller so this is mostly a safety net
)

# The vendored MULTCrossModel hardcodes the text time_attn's input dimension to 768
# (see `self.time_attn = multiTimeAttention(768, self.d_txt, args.embed_time, 8)` in
# MULTCrossModel.__init__ above) -- NOT derived from orig_d_txt, despite orig_d_txt being
# used elsewhere in the same class. This means text_model_name MUST produce 768-dim
# embeddings (standard BERT-base size -- true for Bio_ClinicalBERT, BioBERT, most clinical
# BERT-base checkpoints) or the forward pass will fail with a shape-mismatch error inside
# vendored code we don't control. Checked explicitly in __init__ below rather than left to
# surface as a confusing matmul error three layers deep.
_VENDORED_TEXT_HIDDEN_DIM = 768


def _window_events(hours, extra_tensors, mask, t_hours, lookback_hours, max_events, device):
    """Physically restricts events to [t - lookback_hours, t] and, within that, to the
    max_events MOST RECENT -- not just clamps their time embedding to 0 like the naive
    version did. Events older than lookback_hours already collapse to the identical
    tt=0 representation, so excluding them loses no distinguishable time signal while
    bounding T, which bounds multiTimeAttention's O(B*heads*tt_max*T*value_dim) memory --
    confirmed by direct measurement to reach multiple GB for a single tensor at
    real-data event counts (T~thousands for a long-stay patient's late timepoints),
    which is what caused an OOM kill on the full real-data run this was patched after.

    `extra_tensors` is a list of same-shape [B, T] tensors to re-index identically to
    hours/mask (e.g. var_idx + value for TS, or just left empty for notes since notes'
    "extra" payload is the ragged text list, handled separately by the caller).

    Returns (out_hours, [out_extra, ...], out_mask, kept_indices) where kept_indices is
    a list of B index tensors (into the ORIGINAL T), needed by the text path to also
    subset the ragged text list the same way.
    """
    B, T_in = hours.shape
    age = t_hours.unsqueeze(1) - hours
    keep = mask & (age >= 0) & (age <= lookback_hours)

    counts = keep.sum(dim=1)
    new_T = max(int(counts.max().item()), 1) if B > 0 else 1
    new_T = min(new_T, max_events)

    out_hours = torch.zeros(B, new_T, dtype=hours.dtype, device=device)
    out_mask = torch.zeros(B, new_T, dtype=torch.bool, device=device)
    out_extras = [torch.zeros(B, new_T, dtype=t.dtype, device=device) for t in extra_tensors]
    kept_indices = []

    for b in range(B):
        idx = keep[b].nonzero(as_tuple=True)[0]
        if len(idx) > max_events:
            b_age = age[b, idx]
            order = torch.argsort(b_age)[:max_events]  # smallest age = most recent
            idx = idx[order]
        kept_indices.append(idx)
        n = len(idx)
        if n:
            out_hours[b, :n] = hours[b, idx]
            out_mask[b, :n] = True
            for out_e, in_e in zip(out_extras, extra_tensors):
                out_e[b, :n] = in_e[b, idx]
    return out_hours, out_extras, out_mask, kept_indices


class UTDEBaseline(nn.Module):
    """Contract required by MODEL_REGISTRY: forward(batch) -> logits, shape [B].

    modalities in config controls what actually gets used: include "notes" to get the
    TS+text path (the default model_args below assume this); omit it for TS-only, which
    falls back to a plain linear head on the pooled TS representation (same as this
    file's original TS-only version).
    """

    def __init__(self, config: dict, device: str = "cpu", _tokenizer=None, _biobert=None):
        super().__init__()
        model_args = {**DEFAULT_MODEL_ARGS, **config.get("model_args", {})}
        self.use_text = "notes" in config.get("modalities", ["ts", "notes"])
        if not self.use_text:
            model_args["modeltype"] = "TS"
            model_args["irregular_learn_emb_text"] = False
        self.args = SimpleNamespace(**model_args)
        self.args.model_name = self.args.text_model_name  # BertForRepresentation reads args.model_name
        self.device = device
        self.n_ts_vars = len(VARIABLE_VOCAB)

        biobert = None
        orig_d_txt = None
        if self.use_text:
            if _tokenizer is not None and _biobert is not None:
                # injection point for testing without a network-loaded checkpoint --
                # production use always takes the branch below
                self.tokenizer, biobert = _tokenizer, _biobert
            else:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.text_model_name)
                biobert = AutoModel.from_pretrained(self.args.text_model_name)
            orig_d_txt = biobert.config.hidden_size
            if self.args.freeze_text_encoder:
                for p in biobert.parameters():
                    p.requires_grad = False
                biobert.eval()
            if orig_d_txt != _VENDORED_TEXT_HIDDEN_DIM:
                raise ValueError(
                    f"text_model_name='{self.args.text_model_name}' has hidden_size="
                    f"{orig_d_txt}, but the vendored MULTCrossModel.time_attn hardcodes "
                    f"an input dimension of {_VENDORED_TEXT_HIDDEN_DIM} (see the "
                    f"_VENDORED_TEXT_HIDDEN_DIM comment above) -- pick a BERT-base-sized "
                    f"(768-dim) checkpoint, or this fails deep inside vendored code with "
                    f"a much less clear matmul shape error."
                )

        self.core = MULTCrossModel(
            self.args, device,
            modeltype=self.args.modeltype,
            orig_d_ts=self.n_ts_vars, orig_d_txt=orig_d_txt, Biobert=biobert,
        )
        if not self.use_text:
            # MULTCrossModel only builds its own out_layer under
            # `if modeltype == "TS_Text"` -- the TS-only arm never gets one.
            self.head = nn.Linear(self.args.embed_dim, 1)

    def train(self, mode: bool = True):
        """nn.Module.train() recursively sets ALL submodules to train mode -- without
        this override, a frozen BERT would still have dropout re-enabled every time the
        outer model's .train() is called, even though freeze_text_encoder=True already
        stopped its weights from updating. Pin it to eval mode regardless."""
        super().train(mode)
        if self.use_text and self.args.freeze_text_encoder:
            self.core.bertrep.bert.eval()
        return self

    def forward(self, batch: dict) -> torch.Tensor:
        x_ts, x_ts_mask, ts_tt_list = self._build_ts_inputs(batch["ts"], batch["t_hours"])
        time_query = self.core.learn_time_embedding(self.core.time_query.unsqueeze(0)).to(self.device)
        time_key_ts = self.core.learn_time_embedding(ts_tt_list).to(self.device)
        x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
        x_ts_mask_cat = torch.cat((x_ts_mask, x_ts_mask), 2)
        # [B, tt_max, d_ts] -> [tt_max, B, d_ts], matching TransformerEncoder's
        # seq-first convention (same two lines as MULTCrossModel.forward above)
        proj_x_ts = self.core.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask_cat).transpose(0, 1)
        proj_x_ts = self.core.trans_ts_mem(proj_x_ts) if hasattr(self.core, "trans_ts_mem") else proj_x_ts
        pooled_ts = proj_x_ts[-1]  # last position of the re-aligned reference grid, [B, d_ts]

        if not self.use_text:
            return self.head(pooled_ts).squeeze(-1)

        # NOTE: _build_text_inputs internally re-windows and may produce a smaller
        # effective T than the raw padded shape (e.g. some notes fall outside the
        # lookback window) -- this check on the RAW padded shape just skips the
        # branch entirely when there is obviously nothing to do; _window_events own
        # max(..., 1) floor guarantees _build_text_inputs never receives or returns
        # a truly zero-width tensor either way.
        T = batch["notes"]["hours"].shape[1]
        if T > 0:
            input_ids_seq, attn_mask_seq, note_tt, note_mask = self._build_text_inputs(batch["notes"], batch["t_hours"])
            # KNOWN VENDORED BUG, fixed here rather than in the source file:
            # BertForRepresentation.forward returns torch.stack(txt_arr), which stacks
            # per-note-position [B, hidden] embeddings on a NEW LEADING dim -- i.e.
            # [T, B, hidden], sequence-first. multiTimeAttention.forward immediately
            # does `batch, seq_len, dim = value.size()`, i.e. expects BATCH-first. Fed
            # straight through, T (note count) silently becomes the batch dimension.
            # The transpose(0, 1) below is the actual fix -- confirmed correct by
            # gradient-flow testing, not just shape-checking.
            x_txt = self.core.bertrep(input_ids_seq, attn_mask_seq).transpose(0, 1)  # -> [B, T, hidden]
            time_key_txt = self.core.learn_time_embedding(note_tt).to(self.device)
            proj_x_txt = self.core.time_attn(time_query, time_key_txt, x_txt, note_mask).transpose(0, 1)
            proj_x_txt = self.core.trans_txt_mem(proj_x_txt)
            pooled_txt = proj_x_txt[-1]
        else:
            # defensive only -- collate_sepsis_batch's _pad_stack guarantees T >= 1
            # always, so this branch shouldn't be reachable in practice, but a batch
            # where every sample truly has zero notes (all-False mask) should still
            # produce a sane zero-information fallback rather than break.
            B = pooled_ts.shape[0]
            pooled_txt = torch.zeros(B, self.args.embed_dim, device=self.device)

        # matches the vendored order in MULTCrossModel.forward's else-branch
        # (torch.cat([proj_x_txt[-1], proj_x_ts[-1]], dim=1)) -- cosmetic (a Linear
        # over the concatenation doesn't care about order), kept for fidelity.
        last_hs = torch.cat([pooled_txt, pooled_ts], dim=1)
        last_hs_proj = self.core.proj2(
            F.dropout(F.relu(self.core.proj1(last_hs)), p=self.args.dropout, training=self.training)
        )
        last_hs_proj = last_hs_proj + last_hs
        return self.core.out_layer(last_hs_proj).squeeze(-1)

    def _build_ts_inputs(self, ts: dict, t_hours: torch.Tensor):
        """dataset.py gives an event stream per sample (hours, var_idx, value, mask),
        padded to the batch's max event count -- not the dense [B, T, F] grid this
        model's irregular_learn_emb_ts path wants.

        FIRST windows to lookback_hours + max_ts_events (see _window_events -- this
        bounds T, which bounds multiTimeAttention's memory; without it, a long-stay
        patient's late timepoints can have thousands of raw events and OOM), THEN
        scatters into the dense grid. No binning of VALUES happens -- every kept
        observation still keeps its own exact row and timestamp, only the total count
        is bounded.

        Time normalization is anchored to THIS sample's prediction time t, not to
        admission start -- see forward()'s pooling of proj_x_ts[-1] ("now" = tt=1.0).
        """
        t_hours_dev = t_hours.to(self.device)
        var_idx_raw = ts["var_idx"].clamp(min=0).to(self.device)
        value_raw = ts["value"].to(self.device)
        w_hours, (w_var_idx, w_value), w_mask, _ = _window_events(
            ts["hours"].to(self.device), [var_idx_raw, value_raw], ts["mask"].to(self.device),
            t_hours_dev, self.args.lookback_hours, self.args.max_ts_events, self.device,
        )

        B, T = w_value.shape
        F_dim = self.n_ts_vars
        x = torch.zeros(B, T, F_dim, device=self.device)
        mask = torch.zeros(B, T, F_dim, device=self.device)
        for b in range(B):
            valid = w_mask[b]
            rows = torch.arange(T, device=self.device)[valid]
            cols = w_var_idx[b][valid]
            x[b, rows, cols] = w_value[b][valid]
            mask[b, rows, cols] = 1.0

        age_hours = t_hours_dev.unsqueeze(1) - w_hours
        tt = (1.0 - age_hours / self.args.lookback_hours).clamp(0.0, 1.0)
        return x, mask, tt

    def _build_text_inputs(self, notes: dict, t_hours: torch.Tensor):
        """Same windowing principle as _build_ts_inputs (see _window_events) -- bounds
        the number of notes fed into text_attn to text_lookback_hours + max_notes.
        Note counts are naturally much smaller than TS event counts, so this is more of
        a safety net than the primary fix, but the same unbounded-T memory mechanism
        applies here too if left unguarded.

        dataset.py collate keeps note text as a RAGGED list of B lists (one per sample,
        true length = that sample real note count) aligned with notes hours/mask
        (padded to the batch max note count). _window_events gives us, per sample,
        WHICH ORIGINAL positions survive windowing (kept_indices) -- used here to
        subset the ragged text list identically, since text cannot be gathered by the
        same tensor-indexing trick used for TS numeric value/var_idx.

        For each note POSITION 0..T'-1 (T' = post-windowing count), gather that
        position text across the batch, tokenize as one batch -- the per-position
        batching structure BertForRepresentation.forward expects.

        Time normalization mirrors _build_ts_inputs -- same t-anchoring, separate
        text_lookback_hours knob.
        """
        t_hours_dev = t_hours.to(self.device)
        w_hours, _, w_mask, kept_indices = _window_events(
            notes["hours"].to(self.device), [], notes["mask"].to(self.device),
            t_hours_dev, self.args.text_lookback_hours, self.args.max_notes, self.device,
        )
        B, T = w_hours.shape
        texts = notes["text"]

        input_ids_sequences, attn_mask_sequences = [], []
        for pos in range(T):
            texts_at_pos = []
            for b in range(B):
                if pos < len(kept_indices[b]):
                    orig_pos = kept_indices[b][pos].item()
                    texts_at_pos.append(texts[b][orig_pos] if orig_pos < len(texts[b]) else "")
                else:
                    texts_at_pos.append("")
            enc = self.tokenizer(texts_at_pos, padding=True, truncation=True,
                                  max_length=self.args.max_note_tokens, return_tensors="pt")
            input_ids_sequences.append(enc["input_ids"].to(self.device))
            attn_mask_sequences.append(enc["attention_mask"].to(self.device))

        age_hours = t_hours_dev.unsqueeze(1) - w_hours
        tt = (1.0 - age_hours / self.args.text_lookback_hours).clamp(0.0, 1.0)
        note_mask = w_mask.float()
        return input_ids_sequences, attn_mask_sequences, tt, note_mask