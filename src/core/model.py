import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .models.graph_clf import GraphClf
from .utils import constants as Constants


class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config):
        self.config = config.copy()
        self.net_module = GraphClf

        self.criterion = F.nll_loss
        self.score_func = accuracy
        self.metric_name = 'acc'

        self._init_new_network()

        self._init_optimizer()

    def init_saved_network(self, saved_dir):

        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(
            fname, map_location=lambda storage, loc: storage)
        self.state_dict = saved_params['state_dict']

        self.network = self.net_module(self.config)

        # Merge the arguments
        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

    def _init_new_network(self):
        # this should be creating net_module, not forwarding
        self.network = self.net_module(self.config)

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(
                parameters, lr=self.config['learning_rate'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.config['optimizer'])

    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(
                dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')

    def clip_grad(self):
        if self.config['grad_clipping']:
            parameters = [p for p in self.network.parameters()
                          if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(
                parameters, self.config['grad_clipping'])


def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels)
