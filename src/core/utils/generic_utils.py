import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x


def normalize_adj(mx):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


def normalize_sparse_adj(mx):
    """Row-normalize sparse matrix: symmetric normalized Laplacian"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def to_undirected(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    else:
        num_nodes = max(num_nodes, edge_index.max() + 1)

    row, col = edge_index
    data = np.ones(edge_index.shape[1])
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = (adj + adj.transpose()) > 0
    return adj.astype(np.float64)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
