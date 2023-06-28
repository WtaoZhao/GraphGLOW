import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.constants import VERY_SMALL_NUMBER


def sample_pivots(node_vec, s):
    idx = torch.randperm(node_vec.size(0))[:s]
    return node_vec[idx], idx


def compute_pivot_adj(node_pivot_adj, pivot_mask=None):
    '''Can be more memory-efficient'''
    pivot_node_adj = node_pivot_adj.transpose(-1, -2)
    pivot_norm = torch.clamp(pivot_node_adj.sum(
        dim=-2), min=VERY_SMALL_NUMBER) ** -1
    pivot_adj = torch.matmul(
        pivot_node_adj, pivot_norm.unsqueeze(-1) * node_pivot_adj)

    markoff_value = 0
    if pivot_mask is not None:
        pivot_adj = pivot_adj.masked_fill_(
            1 - pivot_mask.byte().unsqueeze(-1), markoff_value)
        pivot_adj = pivot_adj.masked_fill_(
            1 - pivot_mask.byte().unsqueeze(-2), markoff_value)

    return pivot_adj


class PivotGCNLayer(nn.Module):
    '''
    Adapt GCN layer to perform message passing on node-pivot graph.
    '''
    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super().__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias)

    def forward(self, input, adj, pivot_mp=True, batch_norm=True):
        support = torch.matmul(input, self.weight)

        if pivot_mp:
            node_pivot_adj = adj
            node_norm = node_pivot_adj / \
                torch.clamp(torch.sum(node_pivot_adj, dim=-2,
                            keepdim=True), min=VERY_SMALL_NUMBER)
            pivot_norm = node_pivot_adj / \
                torch.clamp(torch.sum(node_pivot_adj, dim=-1,
                            keepdim=True), min=VERY_SMALL_NUMBER)
            output = torch.matmul(pivot_norm, torch.matmul(
                node_norm.transpose(-1, -2), support))

        else:
            node_adj = adj
            output = torch.matmul(node_adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class PivotGCN(nn.Module):
    '''
    Adapt GCN to perform message passing on node-pivot graph.
    '''
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super().__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(PivotGCNLayer(
            nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(PivotGCNLayer(
                nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(
            PivotGCNLayer(nhid, nclass, batch_norm=False))

    def reset_parameters(self):
        for layer in self.graph_encoders:
            layer.reset_parameters()

    def forward(self, x, node_pivot_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_pivot_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_pivot_adj)

        return x
