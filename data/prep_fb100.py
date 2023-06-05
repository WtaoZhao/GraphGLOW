'''
Preprocess facebook100 datasets.
'''
import argparse
import os
import pickle as pkl

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import label_binarize


def load_fb100(dataset):
    assert dataset in {'amherst', 'cornell', 'jh', 'penn', 'reed'}
    name_map = {
        'amherst': 'Amherst41',
        'cornell': 'Cornell5',
        'jh': 'Johns Hopkins55',
        'penn': 'Penn94',
        'reed': 'Reed98'
    }
    # DATAPATH = '../data/fb/'
    DATAPATH = 'fb/'
    fp = os.path.join(DATAPATH, dataset, name_map[dataset]+'.mat')
    mat = loadmat(fp)
    A = mat['A']  # scipy.sparse.csc.csc_matrix
    metadata = mat['local_info']
    metadata = metadata.astype(np.int)
    num_node = metadata.shape[0]
    print(f'number of nodes:{num_node}')
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))
    # print(features.dtype)  # dtype is float64
    # print(f'label shape:{label.shape}')
    # print(f'feature shape:{features.shape}')
    # print(f'min label value:{np.min(label)}')
    return A, label, features


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = np.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = np.random.permutation(n)

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:

        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def load_save(dataset, out_root='fb'):
    '''
    Save features, labels and adjacency matrix.
    This function can also generate random splits (but I have commented out the code).
    '''
    A, label, features = load_fb100(dataset)

    output_root = os.path.join(out_root, dataset)
    A_path = os.path.join(output_root, 'adj.pkl')
    label_path = os.path.join(output_root, 'labels.pkl')
    feat_path = os.path.join(output_root, 'features.pkl')

    with open(A_path, 'wb') as f1, open(label_path, 'wb') as f2,\
            open(feat_path, 'wb') as f3:
        pkl.dump(A, f1)
        pkl.dump(label, f2)
        pkl.dump(features, f3)

    # idx_train, idx_val, idx_test = rand_train_test_idx(label)
    # print(f'len idx_train:{idx_train.shape}')
    # print(f'len idx_val:{idx_val.shape}')
    # print(f'len idx_test:{idx_test.shape}')

    # train_idx_path = os.path.join(output_root, 'idx_train.pkl')
    # val_idx_path = os.path.join(output_root, 'idx_val.pkl')
    # test_idx_path = os.path.join(output_root, 'idx_test.pkl')
    # with open(train_idx_path, 'wb') as train_f, open(val_idx_path, 'wb') as val_f, \
    #         open(test_idx_path, 'wb') as test_f:
    #     pkl.dump(idx_train, train_f)
    #     pkl.dump(idx_val, val_f)
    #     pkl.dump(idx_test, test_f)


def count_edge(dataset):
    assert dataset in {'amherst', 'cornell', 'jh', 'penn', 'reed'}
    name_map = {
        'amherst': 'Amherst41',
        'cornell': 'Cornell5',
        'jh': 'Johns Hopkins55',
        'penn': 'Penn94',
        'reed': 'Reed98'
    }
    DATAPATH = 'fb/'
    fp = os.path.join(DATAPATH, dataset, name_map[dataset]+'.mat')
    mat = loadmat(fp)
    A = mat['A']  # scipy.sparse.csc.csc_matrix
    num_edge = A.count_nonzero()
    print(f'num of edges:{num_edge}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amherst',
                        choices=['amherst', 'cornell', 'jh', 'reed'])
    args = parser.parse_args()
    load_save(args.dataset)
