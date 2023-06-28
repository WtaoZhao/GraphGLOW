import torch
import torch.nn as nn
import torch.nn.functional as F

from .pivot import PivotGCN


class GraphClf(nn.Module):
    '''
    A wrapper for GNN.
    '''
    def __init__(self, config):
        super(GraphClf, self).__init__()
        self.config = config
        self.name = 'GraphClf'
        self.device = config['device']
        nfeat = config['num_feat']
        nclass = config['num_class']
        hidden_size = config['hidden_size']
        self.dropout = config['dropout']

        self.feature_extractor = nn.Linear(
            in_features=nfeat, out_features=hidden_size)

        gcn_module = PivotGCN
        self.encoder = gcn_module(nfeat=nfeat,
                                  nhid=hidden_size,
                                  nclass=nclass,
                                  graph_hops=config.get('graph_hops', 2),
                                  dropout=self.dropout,
                                  batch_norm=config.get('batch_norm', False))

    def forward(self, node_features, adj):
        node_vec = self.encoder(node_features, adj)
        output = F.log_softmax(node_vec, dim=-1)
        return output
