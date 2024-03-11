from layers import MLP, HalfNLHconv
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn import Parameter


class SetGNN(nn.Module):
    def __init__(self, args, norm=None):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """
        #         V_in_dim = V_dict['in_dim']
        #         V_enc_hid_dim = V_dict['enc_hid_dim']
        #         V_dec_hid_dim = V_dict['dec_hid_dim']
        #         V_out_dim = V_dict['out_dim']
        #         V_enc_num_layers = V_dict['enc_num_layers']
        #         V_dec_num_layers = V_dict['dec_num_layers']

        #         E_in_dim = E_dict['in_dim']
        #         E_enc_hid_dim = E_dict['enc_hid_dim']
        #         E_dec_hid_dim = E_dict['dec_hid_dim']
        #         E_out_dim = E_dict['out_dim']
        #         E_enc_num_layers = E_dict['enc_num_layers']
        #         E_dec_num_layers = E_dict['dec_num_layers']

        #         Now set all dropout the same, but can be different
        self.num_layers = args.GNNs_num_layers
        self.dropout = args.GNNs_dropout
        self.aggr = args.GNNs_aggregate
        self.normalization = args.GNNs_normalization
        self.InputNorm = args.GNNs_input_norm
        self.GPR = args.GNNs_GPR
        self.LearnMask = args.GNNs_LearnMask
        #         Now define V2EConvs[i], V2EConvs[i] for ith layers
        #         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
        #         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        if self.LearnMask:
            self.Importance = Parameter(torch.ones(norm.size()))

        if self.num_layers == 0:
            self.classifier = MLP(
                in_channels=args.GNNs_num_features,
                hidden_channels=args.GNNs_classifier_hidden,
                out_channels=args.GNNs_num_classes,
                num_layers=args.GNNs_num_classifier_layers,
                dropout=args.GNNs_dropout,
                Normalization=args.GNNs_normalization,
                InputNorm=False,
            )
        else:
            self.V2EConvs.append(
                HalfNLHconv(
                    in_dim=args.GNNs_num_features,
                    hid_dim=args.GNNs_MLP_hidden,
                    out_dim=args.GNNs_MLP_hidden,
                    num_layers=args.GNNs_num_MLP_layers,
                    dropout=args.GNNs_dropout,
                    Normalization=args.GNNs_normalization,
                    InputNorm=self.InputNorm,
                    heads=args.GNNs_heads,
                    attention=args.GNNs_PMA,
                )
            )
            self.bnV2Es.append(nn.BatchNorm1d(args.GNNs_MLP_hidden))
            self.E2VConvs.append(
                HalfNLHconv(
                    in_dim=args.GNNs_MLP_hidden,
                    hid_dim=args.GNNs_MLP_hidden,
                    out_dim=args.GNNs_MLP_hidden,
                    num_layers=args.GNNs_num_MLP_layers,
                    dropout=args.GNNs_dropout,
                    Normalization=args.GNNs_normalization,
                    InputNorm=args.GNNs_input_norm,
                    heads=args.GNNs_heads,
                    attention=args.GNNs_PMA,
                )
            )
            self.bnE2Vs.append(nn.BatchNorm1d(args.GNNs_MLP_hidden))
            for _ in range(self.num_layers - 1):
                self.V2EConvs.append(
                    HalfNLHconv(
                        in_dim=args.GNNs_MLP_hidden,
                        hid_dim=args.GNNs_MLP_hidden,
                        out_dim=args.GNNs_MLP_hidden,
                        num_layers=args.GNNs_num_MLP_layers,
                        dropout=self.dropout,
                        Normalization=self.normalization,
                        InputNorm=self.InputNorm,
                        heads=args.GNNs_heads,
                        attention=args.GNNs_PMA,
                    )
                )
                self.bnV2Es.append(nn.BatchNorm1d(args.GNNs_MLP_hidden))
                self.E2VConvs.append(
                    HalfNLHconv(
                        in_dim=args.GNNs_MLP_hidden,
                        hid_dim=args.GNNs_MLP_hidden,
                        out_dim=args.GNNs_MLP_hidden,
                        num_layers=args.GNNs_num_MLP_layers,
                        dropout=self.dropout,
                        Normalization=self.normalization,
                        InputNorm=self.InputNorm,
                        heads=args.GNNs_heads,
                        attention=args.GNNs_PMA,
                    )
                )
                self.bnE2Vs.append(nn.BatchNorm1d(args.GNNs_MLP_hidden))
            if self.GPR:
                self.MLP = MLP(
                    in_channels=args.GNNs_num_features,
                    hidden_channels=args.GNNs_MLP_hidden,
                    out_channels=args.GNNs_MLP_hidden,
                    num_layers=args.GNNs_num_MLP_layers,
                    dropout=self.dropout,
                    Normalization=self.normalization,
                    InputNorm=False,
                )
                self.GPRweights = Linear(self.num_layers + 1, 1, bias=False)
                self.classifier = MLP(
                    in_channels=args.GNNs_MLP_hidden,
                    hidden_channels=args.GNNs_classifier_hidden,
                    out_channels=args.GNNs_num_classes,
                    num_layers=args.GNNs_num_classifier_layers,
                    dropout=self.dropout,
                    Normalization=self.normalization,
                    InputNorm=False,
                )
            else:
                self.classifier = MLP(
                    in_channels=args.GNNs_MLP_hidden,
                    hidden_channels=args.GNNs_classifier_hidden,
                    out_channels=args.GNNs_num_classes,
                    num_layers=args.GNNs_num_classifier_layers,
                    dropout=self.dropout,
                    Normalization=self.normalization,
                    InputNorm=False,
                )

    #         Now we simply use V_enc_hid=V_dec_hid=E_enc_hid=E_dec_hid
    #         However, in general this can be arbitrary.

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, x, edge_index, norm):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """

        if self.LearnMask:
            norm = self.Importance * norm

        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            self.embeddings = x
            x = self.classifier(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr))
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            self.embeddings = x
            x = self.classifier(x)
        return x
