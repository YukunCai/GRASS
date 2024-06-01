from dgl.data import ActorDataset, CoraGraphDataset, PubmedGraphDataset, CornellDataset
from dgl import RowFeatNormalizer
import torch
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import dgl


def get_dataset(name: str = 'cora', raw_dir: str = './data/'):
    if name == 'cora':
        dataset = CoraGraphDataset(raw_dir=raw_dir, transform=RowFeatNormalizer())
    elif name == 'pubmed':
        dataset = PubmedGraphDataset(raw_dir=raw_dir, transform=RowFeatNormalizer())
    elif name == 'actor':
        dataset = ActorDataset(raw_dir=raw_dir, transform=RowFeatNormalizer())
    elif name == 'cornell':
        dataset = CornellDataset(raw_dir=raw_dir, transform=RowFeatNormalizer())
    else:
        raise ValueError('Invalid dataset name: {}'.format(name))
    g = dataset[0]
    feat = g.ndata['feat']
    # ppmi_ = ppmi(feat.numpy())
    # ppmi_ = torch.from_numpy(ppmi_).float()
    label = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    cache_dataset = os.path.exists(
        os.path.join('./cache/', name + '_ppmi.pt')
    )
    if cache_dataset:
        print("Loading cache PPMI Matrix")
        ppmi_matrix = torch.load(os.path.join('./cache/', name + '_ppmi.pt'))
    else:
        print("Begin compute PPMI Matrix")
        edge_index_fea = compute_ppmi(feat.numpy())
        nan_indices = np.isnan(edge_index_fea)
        edge_index_fea[nan_indices] = 0

        ppmi_matrix = MinMaxScaler().fit_transform(edge_index_fea)
        ppmi_matrix = torch.from_numpy(ppmi_matrix).float()
        ppmi_matrix = 0.5 * (ppmi_matrix + ppmi_matrix.t())

        torch.save(edge_index_fea, os.path.join('./cache/', name + '_ppmi.pt'))
        print("Done Compution")
    ppmi_matrix = torch.tensor(ppmi_matrix)
    return g, feat, ppmi_matrix, label, train_mask, val_mask, test_mask


def compute_ppmi(feat_matrix):

    feat_matrix = feat_matrix
    feat_matrix_t = feat_matrix.T

    cooc_matrix = np.matmul(feat_matrix_t, feat_matrix)

    feat_freq = np.array(feat_matrix.sum(axis=0)).reshape(-1)

    total_nodes = feat_matrix.shape[0]

    ppmi_matrix = np.zeros_like(cooc_matrix)

    # 计算PPMI
    for i in range(feat_matrix.shape[1]):
        for j in range(feat_matrix.shape[1]):
            cooc = cooc_matrix[i, j]
            feat_freq_i = feat_freq[i]
            feat_freq_j = feat_freq[j]
            ppmi = max(np.log(((cooc * total_nodes) / (feat_freq_i * feat_freq_j)) + 1e-5), 0)
            # ppmi = np.log((cooc * total_nodes) / (feat_freq_i * feat_freq_j))
            ppmi_matrix[i, j] = ppmi
    ppmi_matrix[ppmi_matrix < 0] = 0.0
    ppmi_matrix[np.isinf(ppmi_matrix)] = 0.0
    ppmi_matrix[np.isnan(ppmi_matrix)] = 0.0
    return ppmi_matrix


def min_max_normalize(matrix):
    matrix_max = np.max(matrix, axis=0)
    matrix_min = np.min(matrix, axis=0)
    matrix_norm = (matrix - matrix_min) / (matrix_max - matrix_min)
    return matrix_norm
