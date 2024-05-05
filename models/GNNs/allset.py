from zmq import Message
from .layers import MLP, HalfNLHconv, AllSetLayer
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn import Parameter

from torch_scatter import scatter


class SetGNN(nn.Module):
    def __init__(self, args, norm=None):
        super(SetGNN, self).__init__()
        self.args = args
        self.num_layers = args.GNNs_num_layers
        self.layers = nn.ModuleList(
            [EncoderLayer(args) for _ in range(self.num_layers)]
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        edge_index = data.edge_index
        x_s, x_t = data.x_s, data.x_t
        num_nodes, num_hyperedges = x_s.size(0), x_t.size(0)
        self_loop = (
            torch.LongTensor([[i, num_hyperedges + i] for i in range(num_nodes)])
            .to(edge_index.device)
            .T
        )
        edge_index = torch.cat([edge_index, self_loop], dim=1)

        emb_V, emb_E = x_s, torch.cat([x_t, x_s], dim=0)
        for i, layer in enumerate(self.layers):
            emb_V, emb_E = layer(emb_V, emb_E, edge_index)
        return emb_V, emb_E[:num_hyperedges]


class Embedding(nn.Module):
    def __init__(self, args):
        super(Embedding, self).__init__()
        self.embed_tokenizor = nn.Embedding(
            args.LLMs_pretrain_vocab_size, args.GNNs_hidden_dim, args.pad_token_id
        )
        self.norm = nn.LayerNorm(args.GNNs_hidden_dim, eps=args.GNNs_layernorm_eps)
        self.dropout = nn.Dropout(args.GNNs_dropout)

    def reset_parameters(self):
        self.embed_tokenizor.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x_s, x_t):
        emb_V, emb_E = self.embed_tokenizor(x_s), self.embed_tokenizor(x_t)
        emb_V = torch.div(
            torch.sum(emb_V, dim=1), torch.count_nonzero(x_s, dim=1).unsqueeze(-1)
        )
        emb_E = torch.div(
            torch.sum(emb_E, dim=1), torch.count_nonzero(x_t, dim=1).unsqueeze(-1)
        )

        emb_V, emb_E = self.norm(emb_V), self.norm(emb_E)
        emb_V, emb_E = self.dropout(emb_V), self.dropout(emb_E)
        return emb_V, emb_E

    def forward_(self, x):
        emb = self.embed_tokenizor(x)
        emb = torch.div(
            torch.sum(emb, dim=1), torch.count_nonzero(x, dim=1).unsqueeze(-1)
        )
        emb = self.norm(emb)
        emb = self.dropout(emb)
        return emb


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.dropout = args.GNNs_dropout
        # self.V2E = AllSetLayer(args)
        # self.E2V = AllSetLayer(args)

        self.V2E = nn.Linear(args.GNNs_hidden_dim, args.GNNs_hidden_dim)
        self.E2V = nn.Linear(args.GNNs_hidden_dim, args.GNNs_hidden_dim)

        self.fuse = nn.Linear(args.GNNs_hidden_dim * 2, args.GNNs_hidden_dim)

    def reset_parameters(self):
        self.fuse.reset_parameters()
        self.V2E.reset_parameters()
        self.E2V.reset_parameters()

    def forward(self, emb_V, emb_E, edge_index):
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        emb_E_tem = self.V2E(emb_V)

        emb_E_tem = F.relu(
            scatter(emb_E_tem[edge_index[0, :]], edge_index[1, :], dim=0, reduce="mean")
        )

        emb_E = torch.cat([emb_E, emb_E_tem], dim=-1)
        emb_E = F.dropout(self.fuse(emb_E), p=self.dropout, training=self.training)

        emb_V = self.E2V(emb_E)
        emb_V = F.relu(
            scatter(
                emb_V[reversed_edge_index[0, :]],
                reversed_edge_index[1, :],
                dim=0,
                reduce="mean",
            )
        )

        emb_V = F.dropout(emb_V, p=self.dropout, training=self.training)
        # emb_E_tem = F.relu(self.V2E(emb_V, edge_index))

        # emb_E = torch.cat([emb_E, emb_E_tem], dim=-1)

        # emb_E = F.dropout(self.fuse(emb_E), p=self.dropout, training=self.training)

        # emb_V = F.relu(self.E2V(emb_E, reversed_edge_index))
        # emb_V = F.dropout(emb_V, p=self.dropout, training=self.training)

        return emb_V, emb_E


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

        self.num_layers = args.GNNs_num_layers

        self.embed_layer = Embedding(args)
        self.layers = nn.ModuleList(
            [EncoderLayer(args) for _ in range(self.num_layers)]
        )

    def reset_parameters(self):
        self.embed_layer.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        emb_V, emb_E = self.embed_layer(data.x_s, data.x_t)
        emb_E = torch.cat([emb_E, emb_V], dim=0)

        edge_index = data.edge_index

        num_nodes = data.x_s.size(0)
        num_hyperedges = data.x_t.size(0)
        self_loop = (
            torch.LongTensor([[i, num_hyperedges + i] for i in range(num_nodes)])
            .to(data.edge_index.device)
            .T
        )
        edge_index = torch.cat([edge_index, self_loop], dim=1)

        for i, layer in enumerate(iterable=self.layers):
            emb_V, emb_E = layer(emb_V, emb_E, edge_index)

        return emb_V, emb_E
