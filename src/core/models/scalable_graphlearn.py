import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.generic_utils import to_cuda
from ..utils.constants import INF


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


class PivotGraphLearner(nn.Module):
    '''
    Learn a node-pivot graph from given node features or learned representations.
    '''
    def __init__(self, input_size, hidden_size, topk=None, epsilon=None, num_pers=16, metric_type='attention', device=None):
        super().__init__()
        self.device = device
        self.topk = topk
        self.epsilon = epsilon
        self.metric_type = metric_type
        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for _ in range(num_pers)])

        elif metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

        elif metric_type == 'weighted_cosine_2':
            self.weight_tensor_1 = torch.Tensor(num_pers, input_size)
            self.weight_tensor_2 = torch.Tensor(num_pers, input_size)
            self.weight_tensor_1 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor_1))
            self.weight_tensor_2 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor_2))

        elif metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])

            self.leakyrelu = nn.LeakyReLU(0.2)

        elif metric_type == 'cosine':
            pass

        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))


    def forward(self, nodes, pivots):
        """
        nodes: (n_node, dim)
        pivots: (n_pivot, dim)

        Returns
        attention: (n_node, n_pivot)
        """
        if self.metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                nodes_fc = torch.relu(self.linear_sims[_](nodes))
                attention += torch.matmul(nodes_fc, nodes_fc.transpose(-1, -2))

            attention /= len(self.linear_sims)
            markoff_value = -INF

        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            # (num_weight_filter,1,dim)

            nodes_fc = nodes.unsqueeze(0) * expand_weight_tensor
            # (num_weight,n_node,dim)
            nodes_norm = F.normalize(nodes_fc, p=2, dim=-1)

            pivots_fc = pivots.unsqueeze(0) * expand_weight_tensor
            # (num_weight,n_pivot,dim)
            pivots_norm = F.normalize(pivots_fc, p=2, dim=-1)

            attention = torch.matmul(nodes_norm, pivots_norm.transpose(-1, -2)).mean(0)
            # (n_node,n_pivot)
            markoff_value = 0

        elif self.metric_type == 'weighted_cosine_2':
            expand_weight_tensor_1 = self.weight_tensor_1.unsqueeze(1)
            expand_weight_tensor_2 = self.weight_tensor_2.unsqueeze(1)
            # (num_weight_filter,1,dim)

            nodes_fc = nodes.unsqueeze(0) * expand_weight_tensor_1
            # (num_weight,n_node,dim)
            nodes_norm = F.normalize(nodes_fc, p=2, dim=-1)
            pivots_fc = pivots.unsqueeze(0) * expand_weight_tensor_2
            # (num_weight,n_pivot,dim)
            pivots_norm = F.normalize(pivots_fc, p=2, dim=-1)
            attention = torch.matmul(nodes_norm, pivots_norm.transpose(-1, -2)).mean(0)
            # (n_node,n_pivot)
            markoff_value = 0

        elif self.metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](nodes)
                a_input2 = self.linear_sims2[_](nodes)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
            markoff_value = -INF


        elif self.metric_type == 'cosine':
            nodes_norm = nodes.div(torch.norm(nodes, p=2, dim=-1, keepdim=True))
            attention = torch.mm(nodes_norm, nodes_norm.transpose(-1, -2)).detach()
            markoff_value = 0

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)

        return attention    # (n_node, pivot_size)

    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = to_cuda((markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), self.device)
        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists