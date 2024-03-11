import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()

        self.num_features = args.GNNs_num_features
        self.num_layers = args.GNNs_num_layers
        self.hidden_dim = args.GNNs_hidden_dim
        self.output_dim = args.GNNs_output_dim
        self.dropout = args.GNNs_dropout

        assert self.num_layers > 2, "Number of layers should be greater than 0"
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(self.num_features, self.hidden_dim))
        for _ in range(self.num_layers - 2):
            self.layers.append(self.hidden_dim, self.hidden_dim)
        self.layers.append(self.hidden_dim, self.output_dim)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, adj = data.x, data.edge_index
        for idx, layer in enumerate(self.layers):
            x = F.relu(layer(x, adj))
            if idx != self.num_layers - 1:
                x = F.dropout(x, self.dropout, training=self.training)
        return x
