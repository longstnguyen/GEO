import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, GATConv, JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GENConv


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def intermediate_forward(self, x, edge_index=None, layer_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature_list(self, x, edge_index=None):
        out_list = []
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out_list.append(x)
        x = self.lins[-1](x)
        return x, out_list


class GCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        save_mem=True,
        use_bn=True,
    ):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def intermediate_forward(self, x, edge_index, layer_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature_list(self, x, edge_index):
        out_list = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out_list.append(x)
        x = self.convs[-1](x, edge_index)
        return x, out_list


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=False,
        heads=2,
        out_heads=1,
    ):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=True)
        )

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    dropout=dropout,
                    heads=heads,
                    concat=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                dropout=dropout,
                heads=out_heads,
                concat=False,
            )
        )

        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def intermediate_forward(self, x, edge_index, layer_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature_list(self, x, edge_index):
        out_list = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out_list.append(x)
        x = self.convs[-1](x, edge_index)
        return x, out_list


class MixHopLayer(nn.Module):
    """Our MixHop layer"""

    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops + 1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x)]
        for j in range(1, self.hops + 1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)


class MixHop(nn.Module):
    """our implementation of MixHop
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, hops=2
    ):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * (hops + 1)))
        for _ in range(num_layers - 2):
            self.convs.append(MixHopLayer(hidden_channels * (hops + 1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels * (hops + 1)))

        self.convs.append(MixHopLayer(hidden_channels * (hops + 1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(out_channels * (hops + 1), out_channels)

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, x, edge_index):
        n = x.size(0)
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        x = self.final_project(x)

        return x


class GCNJK(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        save_mem=True,
        jk_type="max",
    ):
        super(GCNJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == "cat":
            self.final_project = nn.Linear(hidden_channels * num_layers, out_channels)
        else:  # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x


class GATJK(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        heads=2,
        jk_type="max",
    ):
        super(GATJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))

        self.dropout = dropout
        self.activation = F.elu  # note: uses elu

        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels * heads, num_layers=1)
        if jk_type == "cat":
            self.final_project = nn.Linear(hidden_channels * heads * num_layers, out_channels)
        else:  # max or lstm
            self.final_project = nn.Linear(hidden_channels * heads, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x


class GEN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GEN, self).__init__()

        self.dropout = dropout

        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.convs.append(
                GENConv(
                    hidden_channels,
                    hidden_channels,
                    aggr="softmax",
                    t=1.0,
                    learn_t=True,
                    msg_norm=True,
                    learn_msg_scale=True,
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.conv_out = GENConv(
            hidden_channels,
            out_channels,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            msg_norm=True,
            learn_msg_scale=True,
        )
        self.last_norm = nn.BatchNorm1d(hidden_channels)

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.last_norm.reset_parameters()
        self.conv_out.reset_parameters()

    def forward(self, x, edge_index):
        x = self.node_encoder(x)

        for i, conv in enumerate(self.convs):
            x_res = x

            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = conv(x, edge_index)

            x = x + x_res

        x = self.last_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.conv_out(x, edge_index)

        return out

    def feature_list(self, x, edge_index):
        x = self.node_encoder(x)

        for i, conv in enumerate(self.convs):
            x_res = x

            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = conv(x, edge_index)

            x = x + x_res

        x = self.last_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        emb = F.normalize(x, p=2, dim=1)

        out = self.conv_out(x, edge_index)

        return out, [emb]


if __name__ == "__main__":
    a = torch.ones((3, 4))
    print(len(a.shape))
