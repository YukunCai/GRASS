import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import add_self_loops
from loss import info_nce_loss, ce_loss


class recon_feature(nn.Module):
    def __init__(self, num_node, feat_dim, out_dim):
        super(recon_feature, self).__init__()
        self.transA = nn.Parameter(torch.empty(size=(num_node, out_dim)), requires_grad=False)
        self.transT = nn.Parameter(torch.empty(size=(feat_dim, out_dim)), requires_grad=False)

        nn.init.xavier_uniform_(self.transA.data, gain=1.414)
        nn.init.xavier_uniform_(self.transT.data, gain=1.414)

    def forward(self, A, T):
        A = torch.matmul(A, self.transA)
        T = torch.matmul(T, self.transT)
        new_X = torch.concat((A, T), dim=0)
        return new_X, A, T


def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges


def edge_index_to_matrix(edge_index):
    matrix = to_scipy_sparse_matrix(edge_index).todense()
    matrix = torch.tensor(matrix, dtype=torch.float)
    return matrix


class Model(nn.Module):

    def __init__(self,
                 num_node,
                 feat_dim,
                 recon_dim,
                 hidden_dim,
                 decoder_layers,
                 act_fun,
                 dropout,
                 mask=None,
                 loss='ce'
                 ):
        super(Model, self).__init__()
        self.recon_feature = recon_feature(num_node, feat_dim, recon_dim)
        self.mask = mask
        self.dropout = dropout
        self.act_fun = act_fun()
        self.negative_sampler = random_negative_sampler

        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(recon_dim, hidden_dim))
        for _ in range(decoder_layers):
            self.decoder.append(nn.Linear(hidden_dim, hidden_dim))

        if loss == "ce":
            self.loss_fn = ce_loss
        elif loss == "info_nce":
            self.loss_fn = info_nce_loss
        else:
            raise ValueError(loss)
        self.params_1 = list(self.recon_feature.parameters())
        self.params_2 = list(self.decoder.parameters())

    def forward(self, data, adj_dense_matrix, grad_norm=1.0):
        x, self.A, self.T = self.recon_feature(data.matrix, data.T)
        z = torch.matmul(adj_dense_matrix, x)
        edge_index = data.edge_index
        if self.mask != None:
            remaining_edges, masked_edges = self.mask(edge_index)
        aug_edge_index, _ = add_self_loops(edge_index)

        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)

        masked_edges = edge_index_to_matrix(masked_edges).to(z.device)
        neg_edges = edge_index_to_matrix(neg_edges).to(z.device)

        z1 = torch.matmul(masked_edges, x)
        z2 = torch.matmul(neg_edges, x)
        if self.dropout > 0:
            z1 = F.dropout(z1, p=self.dropout, training=True)
            z2 = F.dropout(z2, p=self.dropout, training=True)
        for i, decoder_layer in enumerate(self.decoder[:-1]):
            z1 = self.act_fun(decoder_layer(z1))
            z2 = self.act_fun(decoder_layer(z2))
        z1 = self.decoder[-1](z1)
        z2 = self.decoder[-1](z2)
        loss = self.loss_fn(z, z1) + self.loss_fn(z, z2)
        loss.backward()
        if grad_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
        return loss

    @torch.no_grad()
    def get_emb(self, data, adj_dense_matrix):
        x, self.A, self.T = self.recon_feature(data.matrix, data.T)
        z = torch.matmul(adj_dense_matrix, x)
        return z
