import math
import torch
import torch.nn as nn
from util.attention import MultiHeadAttention
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    def __init__(self, role_heads, args):
        super(TransformerDecoder, self).__init__()
        self.max_seq_len = args.episode_limit + 1

        # def __init__(self, n_heads, att_dim, att_out_dim, soft_temperature, dim_q, dim_k, dim_v):
        self.canSee = 10
        self.N = args.N
        self.role_embedding_dim = args.role_embedding_dim
        self.self_attn_layers1 = MultiHeadAttention(args.N, args.N * args.role_embedding_dim, args.N * args.role_embedding_dim, args.soft_temperature, args.state_dim, args.state_dim, args.state_dim)
        self.self_attn_layers2 = MultiHeadAttention(role_heads, args.role_att_dim, args.att_out_dim, args.soft_temperature, args.role_embedding_dim, args.role_embedding_dim, args.role_embedding_dim)
        self.ln = nn.LayerNorm(args.N * args.role_embedding_dim)
        self.positional_encoding = self._generate_positional_encoding( self.max_seq_len, args.state_dim).to("cuda")

    def _generate_positional_encoding(self, max_seq_len, d_model):
        """ Generate positional encoding matrix """
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tgt, memory, tgt_mask=None):
        batch_size, tgt_len, _ = tgt.size()
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to("cuda")
        # tgt = tgt + torch.unsqueeze(self.positional_encoding, dim=0)[:, :tgt_len, :]
        q_att = self.self_attn_layers1(tgt, tgt, tgt, tgt_mask)
        q_att = self.ln(q_att)
        q_att = q_att.reshape(-1, self.N, self.role_embedding_dim)
        output = self.self_attn_layers2(q_att, memory, memory)
        return output

    def _generate_square_subsequent_mask(self, tgt_len):
        matrix = torch.full((tgt_len, tgt_len), float('-inf'))  # 初始化为负无穷
        for i in range(tgt_len):
            for j in range(max(0, i - self.canSee-1), i):
                matrix[i, j] = 0
        matrix.fill_diagonal_(0)  # 设置主对角线为0
        return matrix

