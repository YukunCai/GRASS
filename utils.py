import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler
def get_edge_index(x,mx)->torch.Tensor:
    mx = mx.numpy()
    x = x.numpy()
    xt = x.T
    r , c = x.shape[0] , x.shape[1]
    d = r + c
    matrix = np.zeros((d, d))
    matrix[0:r,-c:] = x
    matrix[r:,:-c] = xt
    src,dst = np.nonzero(matrix)
    edge_index = torch.tensor([src,dst],dtype=torch.long)
    adj_dense_matrix = torch.tensor(matrix,dtype=torch.float)
    return edge_index,adj_dense_matrix
def get_ppmi_matrix(x , dataname='cora'):
    x = x.numpy()
    cache_dataset = os.path.exists(
        os.path.join('./cache/', dataname + '_ppmi.pt')
    )
    if cache_dataset:
        print(f'loading {dataname} ppmi matrix from cache')
        ppmi_matrix = torch.load(os.path.join('./cache/', dataname + '_ppmi.pt'))
    else:
        print(f'computing {dataname} ppmi matrix...')
        edge_index_fea = compute_ppmi(x)
        nan_indices = np.isnan(edge_index_fea)
        edge_index_fea[nan_indices] = 0
        ppmi_matrix = MinMaxScaler().fit_transform(edge_index_fea)
        ppmi_matrix = torch.from_numpy(ppmi_matrix).float()
        ppmi_matrix = 0.5 * (ppmi_matrix + ppmi_matrix.t())
        torch.save(ppmi_matrix, os.path.join('./cache/', dataname + '_ppmi.pt'))
    ppmi_matrix = torch.tensor(ppmi_matrix,dtype=torch.float)
    return ppmi_matrix
def compute_ppmi(feat_matrix):
    feat_matrix = feat_matrix
    feat_matrix_t = feat_matrix.T
    cooc_matrix = np.matmul(feat_matrix_t, feat_matrix)
    feat_freq = np.array(feat_matrix.sum(axis=0)).reshape(-1)
    total_nodes = feat_matrix.shape[0]
    ppmi_matrix = np.zeros_like(cooc_matrix)
    for i in range(feat_matrix.shape[1]):
        for j in range(feat_matrix.shape[1]):
            cooc = cooc_matrix[i, j]
            feat_freq_i = feat_freq[i]
            feat_freq_j = feat_freq[j]
            ppmi = max(np.log(((cooc * total_nodes) / (feat_freq_i * feat_freq_j)) + 1e-5), 0)
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
