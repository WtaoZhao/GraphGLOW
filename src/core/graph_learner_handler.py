import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from .models.scalable_graphlearn import PivotGraphLearner


class GL_handler(nn.Module):
    '''
    High level graph structure learner handler that creates structure learner 
    and calls corresponding interface to learn graph structures.
    '''
    def __init__(self, config, save_dir):
        super(GL_handler, self).__init__()
        self.config = config
        self.graph_metric_type = config['graph_metric_type']
        self.graph_skip_conn = config['graph_skip_conn']
        hidden_size = config['hidden_size']

        if not config['no_cuda'] and torch.cuda.is_available():
            self.device = torch.device(
                'cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.dir = os.path.join(save_dir, 'GL.pt')

        graph_learn_fun = PivotGraphLearner
        if config.get('use_both_metric', False):
            self.graph_learner1 = graph_learn_fun(hidden_size,
                                                  config['graph_learn_hidden_size'],
                                                  topk=config['graph_learn_topk'],
                                                  epsilon=config['graph_learn_epsilon'],
                                                  num_pers=config['graph_learn_num_pers'],
                                                  metric_type='weighted_cosine',
                                                  device=self.device)
            self.graph_learner2 = graph_learn_fun(hidden_size,
                                                  config['graph_learn_hidden_size'],
                                                  topk=config['graph_learn_topk'],
                                                  epsilon=config['graph_learn_epsilon'],
                                                  num_pers=config['graph_learn_num_pers'],
                                                  metric_type='weighted_cosine_2',
                                                  device=self.device)
            self.coef1 = torch.randn(1, requires_grad=True, device=self.device)
            self.coef2 = torch.randn(1, requires_grad=True, device=self.device)
        else:
            self.graph_learner = graph_learn_fun(hidden_size,
                                                 config['graph_learn_hidden_size'],
                                                 topk=config['graph_learn_topk'],
                                                 epsilon=config['graph_learn_epsilon'],
                                                 num_pers=config['graph_learn_num_pers'],
                                                 metric_type=config['graph_metric_type'],
                                                 device=self.device)

    def learn_graph(self,  node_features, pivot_features=None):

        if self.config.get('use_both_metric', False):
            node_pivot_adj1 = self.graph_learner1(
                node_features, pivot_features)
            node_pivot_adj2 = self.graph_learner2(
                node_features, pivot_features)
            return self.coef1*node_pivot_adj1+self.coef2*node_pivot_adj2
        node_pivot_adj = self.graph_learner(node_features, pivot_features)
        return node_pivot_adj

    def forward(self, node_features, init_adj=None):
        '''
        learner_no: 1 or 2
        '''
        node_features = F.dropout(node_features, self.config.get(
            'feat_adj_dropout', 0), training=self.training)
        learner = self.graph_learner
        raw_adj, adj = self.learn_graph(learner, node_features)
        adj = F.dropout(adj, self.config.get(
            'feat_adj_dropout', 0), training=self.training)

        return adj

    def clip_grad(self):
        # Clip gradients
        if self.config['grad_clipping']:
            parameters = [p for p in self.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(
                parameters, self.config['grad_clipping'])

    def save(self):
        torch.save(self.state_dict(), self.dir)

    def init_saved_network(self):
        self.load_state_dict(torch.load(
            self.dir, map_location=lambda storage, loc: storage))
