from turtle import forward
import torch, math
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, Size, OptTensor, SparseTensor
from torch_scatter import scatter_add, scatter
from typing import Optional
import torch.nn as nn


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class PMA(MessagePassing):
    """
    PMA part:
    Note that in original PMA, we need to compute the inner product of the seed and neighbor nodes.
    i.e. e_ij = a(Wh_i,Wh_j), where a should be the inner product, h_i is the seed and h_j are neightbor nodes.
    In GAT, a(x,y) = a^T[x||y]. We use the same logic.
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels,
        hid_dim,
        out_channels,
        num_layers,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        bias=False,
        **kwargs,
    ):
        #         kwargs.setdefault('aggr', 'add')
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = 0.0
        self.aggr = "add"
        #         self.input_seed = input_seed

        #         This is the encoder part. Where we use 1 layer NN (Theta*x_i in the GATConv description)
        #         Now, no seed as input. Directly learn the importance weights alpha_ij.
        #         self.lin_O = Linear(heads*self.hidden, self.hidden) # For heads combining
        # For neighbor nodes (source side, key)
        self.lin_K = Linear(in_channels, self.heads * self.hidden)
        # For neighbor nodes (source side, value)
        self.lin_V = Linear(in_channels, self.heads * self.hidden)
        self.att_r = Parameter(torch.Tensor(1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(
            in_channels=self.heads * self.hidden,
            hidden_channels=self.heads * self.hidden,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=0.0,
            Normalization="None",
        )
        self.ln0 = nn.LayerNorm(self.heads * self.hidden)
        self.ln1 = nn.LayerNorm(self.heads * self.hidden)
        #         if bias and concat:
        #             self.bias = Parameter(torch.Tensor(heads * out_channels))
        #         elif bias and not concat:
        #             self.bias = Parameter(torch.Tensor(out_channels))
        #         else:

        #         Always no bias! (For now)
        self.register_parameter("bias", None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        #         glorot(self.lin_l.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        #         glorot(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    #         zeros(self.bias)

    def forward(
        self, x, edge_index: Adj, size: Size = None, return_attention_weights=None
    ):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.hidden

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in `GATConv`."
            x_K = self.lin_K(x).view(-1, H, C)
            x_V = self.lin_V(x).view(-1, H, C)
            alpha_r = (x_K * self.att_r).sum(dim=-1)
        #         else:
        #             x_l, x_r = x[0], x[1]
        #             assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
        #             x_l = self.lin_l(x_l).view(-1, H, C)
        #             alpha_l = (x_l * self.att_l).sum(dim=-1)
        #             if x_r is not None:
        #                 x_r = self.lin_r(x_r).view(-1, H, C)
        #                 alpha_r = (x_r * self.att_r).sum(dim=-1)

        #         assert x_l is not None
        #         assert alpha_l is not None

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        #         ipdb.set_trace()
        out = self.propagate(edge_index, x=x_V, alpha=alpha_r, aggregate=self.aggr)

        alpha = self._alpha
        self._alpha = None

        #         Note that in the original code of GMT paper, they do not use additional W^O to combine heads.
        #         This is because O = softmax(QK^T)V and V = V_in*W^V. So W^O can be effectively taken care by W^V!!!
        out += self.att_r  # This is Seed + Multihead
        # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection. Lhs of eq(7) in GMT paper.
        out = self.ln1(out + F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(self, x_j, alpha_j, index, ptr, size_j):
        #         ipdb.set_trace()
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max() + 1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index, dim_size=None, aggregate=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        if aggregate is None:
            raise ValueError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggregate)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class HalfNLHconv(MessagePassing):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers,
        dropout,
        Normalization="bn",
        InputNorm=False,
        heads=1,
        attention=True,
    ):
        super(HalfNLHconv, self).__init__()

        self.attention = attention
        self.dropout = dropout

        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(
                    in_dim,
                    hid_dim,
                    hid_dim,
                    num_layers,
                    dropout,
                    Normalization,
                    InputNorm,
                )
                self.f_dec = MLP(
                    hid_dim,
                    hid_dim,
                    out_dim,
                    num_layers,
                    dropout,
                    Normalization,
                    InputNorm,
                )
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()

    #         self.bn = nn.BatchNorm1d(dec_hid_dim)
    #         self.dropout = dropout
    #         self.Prop = S2SProp()

    def reset_parameters(self):
        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ == "Identity"):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ == "Identity"):
                self.f_dec.reset_parameters()

    #         self.bn.reset_parameters()

    def forward(self, x, edge_index, norm, aggr="add"):
        """
        input -> MLP -> Prop
        """

        if self.attention:
            x = self.prop(x, edge_index)
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(edge_index, x=x, norm=norm, aggregate=aggr)
            x = F.relu(self.f_dec(x))
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None, aggregate=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        if aggregate is None:
            raise ValueError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggregate)


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout=0.5,
        Normalization="bn",
        InputNorm=False,
    ):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ["bn", "ln", "None"]
        if Normalization == "bn":
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == "ln":
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == "Identity"):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class AllSetLayer(MessagePassing):
    def __init__(self, args, negative_slope=0.2):
        super(AllSetLayer, self).__init__(node_dim=0)

        self.in_channels = args.GNNs_hidden_dim
        self.heads = args.GNNs_num_heads
        self.hidden = args.GNNs_hidden_dim // self.heads
        self.out_channels = args.GNNs_hidden_dim

        self.negative_slope = negative_slope
        self.dropout = args.GNNs_dropout
        self.aggr = "mean"

        self.lin_K = Linear(self.in_channels, self.heads * self.hidden)
        self.lin_V = Linear(self.in_channels, self.heads * self.hidden)
        self.att_r = Parameter(torch.Tensor(1, self.heads, self.hidden))  # Seed vector
        self.rFF = PositionwiseFFN(args)

        self.ln0 = nn.LayerNorm(self.heads * self.hidden)
        self.ln1 = nn.LayerNorm(self.heads * self.hidden)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index):
        H, C = self.heads, self.hidden
        alpha_r: OptTensor = None

        assert x.dim() == 2, "Static graphs not supported in `GATConv`."
        x_K = self.lin_K(x).view(-1, H, C)
        x_V = self.lin_V(x).view(-1, H, C)
        alpha_r = (x_K * self.att_r).sum(dim=-1)

        out = self.propagate(edge_index, x=x_V, alpha=alpha_r, aggregate="add")

        alpha = self._alpha
        self._alpha = None
        out += self.att_r  # Seed + Multihead
        # concat heads then LayerNorm.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection.
        out = self.ln1(out + F.relu(self.rFF(out)))
        return out

    def message(self, x_j, alpha_j, index, ptr, size_j):
        #         ipdb.set_trace()
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max() + 1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index, dim_size=None, aggregate=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        if aggregate is None:
            raise ValueError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggregate)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class PositionwiseFFN(nn.Module):
    """The Position-wise FFN layer used in Transformer-like architectures

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))
    Also, if we use gated projection. We will use
        fc1_1 * act(fc1_2(data)) to map the data
    """

    def __init__(self, args):
        """
        Parameters
        ----------
        units
        hidden_size
        activation_dropout
        dropout
        activation
        normalization
            layer_norm or no_norm
        layer_norm_eps
        pre_norm
            Pre-layer normalization as proposed in the paper:
            "[ACL2018] The Best of Both Worlds: Combining Recent Advances in
             Neural Machine Translation"
            This will stabilize the training of Transformers.
            You may also refer to
            "[Arxiv2020] Understanding the Difficulty of Training Transformers"
        """
        super().__init__()
        self.args = args
        self.dropout_layer = nn.Dropout(args.GNNs_dropout)
        self.activation_dropout_layer = nn.Dropout(args.GNNs_dropout)
        self.ffn_1 = nn.Linear(
            in_features=args.GNNs_hidden_dim,
            out_features=args.GNNs_MLP_hidden,
            bias=True,
        )
        if args.GNNs_gated_proj:
            self.ffn_1_gate = nn.Linear(
                in_features=args.GNNs_hidden_dim,
                out_features=args.GNNs_hidden_dim,
                bias=True,
            )
        self.activation = get_activation(args.GNNs_activation_fn)
        self.ffn_2 = nn.Linear(
            in_features=args.GNNs_MLP_hidden,
            out_features=args.GNNs_hidden_dim,
            bias=True,
        )
        self.layer_norm = nn.LayerNorm(
            eps=args.GNNs_layernorm_eps,
            normalized_shape=args.GNNs_hidden_dim,
        )
        self.init_weights()

    def init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data):
        """

        Parameters
        ----------
        data :
            Shape (B, seq_length, C_in)

        Returns
        -------
        out :
            Shape (B, seq_length, C_out)
        """
        residual = data
        if self.args.GNNs_pre_norm:
            data = self.layer_norm(data)
        if self.args.GNNs_gated_proj:
            out = self.activation(self.ffn_1_gate(data)) * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self.args.GNNs_pre_norm:
            out = self.layer_norm(out)
        return out


def get_activation(act, inplace=False):
    """

    Parameters
    ----------
    act
        Name of the activation
    inplace
        Whether to perform inplace activation

    Returns
    -------
    activation_layer
        The activation
    """
    if act is None:
        return lambda x: x

    if isinstance(act, str):
        if act == "leaky":
            # TODO(sxjscience) Add regex matching here to parse `leaky(0.1)`
            return nn.LeakyReLU(0.1, inplace=inplace)
        if act == "identity":
            return nn.Identity()
        if act == "elu":
            return nn.ELU(inplace=inplace)
        if act == "gelu":
            return nn.GELU()
        if act == "relu":
            return nn.ReLU()
        if act == "sigmoid":
            return nn.Sigmoid()
        if act == "tanh":
            return nn.Tanh()
        if act in {"softrelu", "softplus"}:
            return nn.Softplus()
        if act == "softsign":
            return nn.Softsign()
        raise NotImplementedError(
            'act="{}" is not supported. '
            "Try to include it if you can find that in "
            "https://pytorch.org/docs/stable/nn.html".format(act)
        )

    return act
