import torch.nn as nn
import torch
import torch.nn.functional as F


# 定义Transformer的Encoder部分
class TransformerEncoderModel(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0):
        super(TransformerEncoderModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src, mask=None):
        # 调整输入张量维度使其符合Transformer的要求
        src = src.transpose(0, 1)  # 将第0维度（batch）和第1维度（sequence）交换位置
        output = self.transformer_encoder(src, mask)
        output = output.transpose(0, 1)  # 恢复维度
        return output


class Q_network_RNN(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size*N,input_dim)
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        Q = self.fc2(self.rnn_hidden)
        return Q


class Q_network_MLP(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_MLP, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size,max_episode_len,N,input_dim)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q


class Agent_Embedding(nn.Module):
    def __init__(self, args):
        super(Agent_Embedding, self).__init__()
        self.input_dim = args.obs_dim + args.action_dim
        self.agent_embedding_dim = args.agent_embedding_dim

        self.fc1 = nn.Linear(self.input_dim, self.agent_embedding_dim)
        self.rnn_hidden = None
        self.agent_embedding_fc = nn.GRUCell(self.agent_embedding_dim, self.agent_embedding_dim)
        self.fc2 = nn.Linear(self.agent_embedding_dim, self.agent_embedding_dim)

    def forward(self, obs, last_a, detach=False):
        inputs = torch.cat([obs, last_a], dim=-1)
        fc1_out = torch.relu(self.fc1(inputs))
        self.rnn_hidden = self.agent_embedding_fc(fc1_out, self.rnn_hidden)
        fc2_out = self.fc2(self.rnn_hidden)
        if detach:
            fc2_out.detach()
        return fc2_out


class Agent_Embedding_Decoder(nn.Module):
    def __init__(self, args):
        super(Agent_Embedding_Decoder, self).__init__()
        self.agent_embedding_dim = args.agent_embedding_dim
        self.decoder_out_dim = args.obs_dim + args.N  # out_put: o(t+1)+agent_idx

        self.fc1 = nn.Linear(self.agent_embedding_dim, self.agent_embedding_dim)
        self.fc2 = nn.Linear(self.agent_embedding_dim, self.decoder_out_dim)

    def forward(self, agent_embedding):
        fc1_out = torch.relu(self.fc1(agent_embedding))
        decoder_out = self.fc2(fc1_out)
        return decoder_out


class Role_Embedding(nn.Module):
    def __init__(self, args):
        super(Role_Embedding, self).__init__()
        self.agent_embedding_dim = args.agent_embedding_dim
        self.role_embedding_dim = args.role_embedding_dim
        self.use_ln = args.use_ln

        if self.use_ln:  # 使用layer_norm
            self.role_embeding = nn.Sequential(
                TransformerEncoderModel(num_layers=1, d_model=self.agent_embedding_dim, nhead=4),
                nn.Linear(self.agent_embedding_dim, self.role_embedding_dim),
                nn.LayerNorm(self.role_embedding_dim))
        else:
            self.role_embeding = nn.Sequential(
                TransformerEncoderModel(num_layers=1, d_model=self.agent_embedding_dim, nhead=4),
                nn.Linear(self.agent_embedding_dim, self.role_embedding_dim))

    def forward(self, agent_embedding, detach=False):
        if agent_embedding.dim() < 3:
            agent_embedding = torch.unsqueeze(agent_embedding, dim=0)
            if self.use_ln:
                output = self.role_embeding(agent_embedding)
            else:
                output = self.role_embeding(agent_embedding)
            output = torch.squeeze(output, dim=0)
        else:
            if self.use_ln:
                output = self.role_embeding(agent_embedding)
            else:
                output = self.role_embeding(agent_embedding)
        if detach:
            output.detach()
        output = torch.sigmoid(output)
        return output