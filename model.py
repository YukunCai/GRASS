import dgl
import torch
from scipy.sparse import coo_matrix
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss import info_nce_loss, ce_loss, log_rank_loss, hinge_auc_loss, auc_loss
from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class ParametrisedAdjacencyTransformer(nn.Module):
    def __init__(self, adj_dim, T_dim, d):
        super(ParametrisedAdjacencyTransformer, self).__init__()
        self.W1 = nn.Parameter(torch.Tensor(adj_dim, d))
        nn.init.xavier_uniform_(self.W1)
        self.W2 = nn.Parameter(torch.Tensor(T_dim, d))
        nn.init.xavier_uniform_(self.W2)

    def forward(self, A, T):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if A.dtype != torch.float32:
            A = A.float()
        A = A.to(device)
        if T.dtype != torch.float32:
            T = T.float()
        self.W1 = torch.nn.Parameter(self.W1.to(device).detach())
        self.W2 = torch.nn.Parameter(self.W2.to(device).detach())

        return torch.matmul(A, self.W1), torch.matmul(T, self.W2)


class transform(torch.nn.Module):
    def __init__(self, encoder: ParametrisedAdjacencyTransformer):
        super(transform, self).__init__()
        self.encoder = encoder

    def forward(self, A, T) -> (torch.Tensor, torch.Tensor):
        transform_A, transform_T = self.encoder(A, T)
        return transform_A, transform_T


def build_edge_index(A):
    zero_indices = torch.nonzero(A == 0, as_tuple=False)
    edge_index = torch.zeros(2, zero_indices.size(0), dtype=torch.long)
    edge_index[0] = zero_indices[:, 0]
    edge_index[1] = zero_indices[:, 1]
    return edge_index


def create_graph_from_adjacency_matrix(A_prime):
    if A_prime.dim() != 2 or A_prime.size(0) != A_prime.size(1):
        raise ValueError("邻接矩阵 A' 必须是方阵。")
    rows, cols = torch.nonzero(A_prime, as_tuple=False).t()
    rows, cols = rows[rows != cols], cols[rows != cols]
    edge_index = torch.stack([rows, cols], dim=0).t()
    return edge_index


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "lrelu":
        return nn.LeakyReLU()
    elif activation == "prelu":
        return nn.PReLU()
    else:
        raise ValueError("Unknown activation")


def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 2):  # 两层GCN
        super(Encoder, self).__init__()

        assert k >= 2
        self.k = k
        self.conv = [GCNConv(in_channels, 2 * out_channels)]  # 第一层输入和输出
        for _ in range(1, k - 1):
            self.conv.append(GCNConv(2 * out_channels, 2 * out_channels))  # 隐藏层
        self.conv.append(GCNConv(2 * out_channels, out_channels))  # 隐藏层和输出
        self.conv = nn.ModuleList(self.conv)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = F.relu(self.conv[i](x, edge_index))
        return x


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, activation):
        super(EdgeDecoder, self).__init__()
        self.mlps = nn.ModuleList()
        self.activation = activation
        for i in range(num_layers):
            first_channels = hidden_dim
            second_channels = 1 if i == num_layers - 1 else hidden_dim
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        # self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):

        x = z[edge[0]] * z[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)

        h = self.mlps[-1](x)
        # x = self.activation(x)
        if sigmoid:
            return x.sigmoid()
        else:
            return h


class Decoder_Model(torch.nn.Module):
    def __init__(self, decoder: EdgeDecoder):
        super(Decoder_Model, self).__init__()
        self.decoder: EdgeDecoder = decoder

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, edge_index)

    def loss(self, pos_z, neg_z):
        epsilon = 1e-15
        pos_z = np.clip(pos_z.detach().numpy(), epsilon, 1.0 - epsilon)

        pos_loss = - np.sum(np.log(pos_z), axis=1)
        pos_loss = np.mean(pos_loss)

        neg_z = np.clip(neg_z.detach().numpy(), epsilon, 1.0 - epsilon)
        neg_1 = 1 - neg_z
        neg = np.where(neg_1 != 0, np.log(neg_1), 0)
        neg_loss = - np.sum(neg, axis=1)
        neg_loss = np.mean(neg_loss)
        loss = pos_loss + neg_loss
        return loss
