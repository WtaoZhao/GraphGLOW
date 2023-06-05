import argparse
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import yaml
from core.trans_gnn import trainer


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config):
    print_config(config)
    my_trainer = trainer(config)
    my_trainer.joint_train()  # train on source graphs.
    my_trainer.transfer()  # train on target graphs


def multi_run_main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    configs = grid(config)
    for cnf in configs:
        my_trainer = trainer(cnf)
        my_trainer.joint_train()
        my_trainer.transfer()


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True,
                        type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true',
                        help='flag: multi run')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************\n")


def grid(kwargs):
    '''
    Builds a mesh grid with given keyword arguments for this Config class.
    '''

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        '''
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        '''
        from functools import reduce

        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()),
                   dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i]
         for i, k in enumerate(sin)}
    ) for vv in grd]


if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])

    if cfg['multi_run']:
        multi_run_main(config)
    else:
        main(config)
