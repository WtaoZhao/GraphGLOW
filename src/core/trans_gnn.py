import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

from .graph_learner_handler import GL_handler
from .model_handler import ModelHandler

params = ['seed', 'num_pivots', 'hidden_size', 'graph_metric_type', 'graph_skip_conn',
          'update_adj_ratio', 'smoothness_ratio', 'degree_ratio', 'sparsity_ratio', 'graph_learn_num_pers', 'learning_rate', 'weight_decay', 'max_iter', 'eps_adj', 'max_episodes', 'graph_learn_regularization', 'prob_del_edge', 'dropout', 'gl_dropout', 'feat_adj_dropout']


class trainer:
    '''
    Initialize models and coordinates the training and transferring process.
    '''

    def __init__(self, config):
        self.device = get_device(config)

        # use dropout value if feat_adj_dropout<0 or gl_dropout<0
        if config.get('feat_adj_dropout', -1) < 0:
            config['feat_adj_dropout'] = config['dropout']
        if config.get('gl_dropout', -1) < 0:
            config['gl_dropout'] = config['dropout']

        dataset_name_list = config['dataset_name'].split('+')

        if 'trans_dataset_name' in config:
            trans_dataset_name_list = config.get(
                'trans_dataset_name', '').split('+')
        else:
            trans_dataset_name_list = []

        print_str = config['dataset_name']
        if config.get('trans_dataset_name', '') != '':
            print_str += '_trans_'+config.get('trans_dataset_name', '')

        log_folder = 'results/'
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)
        log_path = log_folder+print_str+'.txt'
        self.log_f = open(log_path, 'a')
        self.log_f.write(f'=============={print_str}================\n')
        print(f'=============={print_str}================')

        model_folder = f'results/{print_str}'
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        param_str = ''
        for key in params:
            if key in config:
                param_str += f'{key}:{config[key]}, '
        self.log_f.write(f'{param_str}\n')

        self.logger = None
        self.config = config
        self.model_handler_list = []
        self.trans_model_handler_list = []

        for dataset in dataset_name_list:
            self.model_handler_list.append(ModelHandler(
                self.config,  dataset_name=dataset, log_f=self.log_f, save_dir=model_folder))

        for dataset in trans_dataset_name_list:
            self.trans_model_handler_list.append(ModelHandler(
                self.config, dataset_name=dataset, log_f=self.log_f, save_dir=model_folder))

        self.gl_handler = GL_handler(self.config, save_dir=model_folder)
        self.gl_handler.to(self.device)
        self.opt_gl_handler = init_optimizer(self.gl_handler, config)

    def joint_train(self):
        max_episodes = self.config.get('max_episodes', 1)
        for episode in range(max_episodes):
            episode_str = f'==================episode[{episode + 1}/{max_episodes}]=========================='
            print(episode_str + '\n')
            self.log_f.write(episode_str + '\n')
            for model_handler in self.model_handler_list:
                format_str = f'{model_handler.dataset}, '
                print(format_str)
                self.log_f.write(format_str)
                model_handler.train(
                    self.gl_handler, self.opt_gl_handler, write_to_log=False)
                model_handler.test(self.gl_handler, self.opt_gl_handler)

        test_str = '==========testing on all datasets====================='
        print(test_str)
        for model_handler in self.model_handler_list:
            test_str = f'testing on {model_handler.dataset}'
            print(test_str)
            self.log_f.write(test_str + ', ')
            model_handler.test(self.gl_handler, self.opt_gl_handler)

    def transfer(self):
        if not self.trans_model_handler_list:
            self.log_f.write('\n\n')
            self.log_f.close()
            return
        for model_handler in self.trans_model_handler_list:
            format_str = f'transferring on {model_handler.dataset}'
            print(format_str)
            model_handler.train(
                self.gl_handler, opt_gl_handler=None, train_gl=False, write_to_log=False)
        test_str = '==========testing on all transfer datasets====================='
        print(test_str)
        for model_handler in self.trans_model_handler_list:
            test_str = f'testing on {model_handler.dataset} transfer dataset'
            print(test_str)
            self.log_f.write(test_str+', ')
            model_handler.test(self.gl_handler, opt_gl_handler=None)
        self.log_f.write('\n\n')
        self.log_f.close()
        return


def init_optimizer(model, config):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(parameters, config['learning_rate'],
                              momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(parameters, lr=config['learning_rate'],
                               weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamax':
        optimizer = optim.Adamax(parameters, lr=config['learning_rate'])
    else:
        raise RuntimeError('Unsupported optimizer: %s' % config['optimizer'])

    return optimizer


def get_device(config):
    if not config['no_cuda'] and torch.cuda.is_available():
        device = torch.device(
            'cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    return device
