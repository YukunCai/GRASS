import dgl
import torch
import torch.nn as nn
import argparse
from utils import get_dataset
from torch.optim import Adam
from GCL.eval import get_split
from eval import LREvaluator
from mask import mask_edges
from model import EdgeDecoder, transform, create_graph_from_adjacency_matrix, Decoder_Model, build_edge_index, \
    ParametrisedAdjacencyTransformer
import warnings
import numpy as np

warnings.filterwarnings("ignore")

def run(model, opt_w1, opt_w2, opt, Z, edge_index, edge_index1):
    model.train()
    for i in range(args.num_epochs):
        model.train()
        opt.zero_grad()
        opt_w1.zero_grad()
        opt_w2.zero_grad()
        pos_z = model(Z, edge_index)
        neg_z = model(Z, edge_index1)
        loss = model.loss(pos_z, neg_z)
        loss_tensor = torch.tensor(loss, requires_grad=True)
        loss_tensor.backward()
        opt.step()
        opt_w1.step()
        opt_w2.step()

def train(args, Z, model_z, edge_index, edge_index1):
    activation = ({'relu': nn.ReLU, 'prelu': nn.PReLU, 'lrelu': nn.LeakyReLU, 'elu': nn.ELU})[args.activation]
    edge_decoder = EdgeDecoder(args.num_hidden, num_layers=args.decoder_layers, dropout=args.dropout, activation=activation)
    model = Decoder_Model(edge_decoder)
    opt_w1 = Adam([model_z.W1], lr=args.lr_1, weight_decay=args.weight_decay)
    opt_w2 = Adam([model_z.W2], lr=args.lr_2, weight_decay=args.weight_decay)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run(model, opt_w1, opt_w2, opt, Z, edge_index, edge_index1)
    model.eval()
    z1 = Z[:feat.shape[0], :]
    z2 = Z[-feat.shape[1]:, :]
    z1 = z1.to(device)
    z2 = z2.to(device)
    z = torch.cat([z1, torch.matmul(feat, z2)], dim=1)
    split = get_split(num_samples=z.size(0), train_ratio=0.48, test_ratio=0.2)
    result = LREvaluator()(z, label, split)
    return result.values()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument('--dataset', type=str, default='actor')
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--num_epochs', type=int, default=100)
    # optim
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_1', type=float, default=0.001)
    parser.add_argument('--lr_2', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    # parser.add_argument('--wd_2', type=float, default=1e-5)
    # model
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_hidden', type=int, default=1024)
    parser.add_argument('--activation', type=str, default='relu')

    # hyper parameters
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')
    parser.add_argument("--start", nargs="?", default="node",
                        help="Which Type to sample starting nodes for random walks, (default: node)")
    parser.add_argument("--decoder_layers", type=int, default=1)
    args = parser.parse_args()

    device = 'cuda'
    g, feat, T, label, train_mask, val_mask, test_mask = get_dataset(args.dataset)
    g = g.to(device)
    feat = feat.to(device)
    T = T.to(device)
    label = label.to(device)
    print(T.device)
    adj_matrix = g.adjacency_matrix()
    dense_adj_matrix = adj_matrix.to_dense()

    model_z = ParametrisedAdjacencyTransformer(dense_adj_matrix.shape[0], T.shape[0], args.num_hidden).to(device)
    transformer = transform(model_z)
    Za, Zf = transformer(dense_adj_matrix, T)
    Za_cpu = Za.detach().cpu().numpy()
    Zf_cpu = Zf.detach().cpu().numpy()
    X_prime = np.vstack((Za_cpu, Zf_cpu))
    X_prime = torch.tensor(X_prime)
    X_row = feat.shape[0]
    X_col = feat.shape[1]
    zero_matrix_n = np.zeros((X_row, X_row))
    zero_matrix_d = np.zeros((X_col, X_col))
    X_transpose = feat.T
    feat_cpu = feat.detach().cpu().numpy()
    X_transpose_cpu = X_transpose.detach().cpu().numpy()
    A_prime = np.vstack((np.hstack([zero_matrix_n, feat_cpu]), np.hstack([X_transpose_cpu, zero_matrix_d])))  # 新图的A
    A_prime_tensor = torch.from_numpy(A_prime)
    A_prime = A_prime_tensor.float()

    edge_index = create_graph_from_adjacency_matrix(A_prime_tensor)
    edge_index1 = build_edge_index(A_prime)
    edge_index1 = edge_index1.t()

    mask_A = mask_edges(A_prime, p=args.p)

    sub_matrix = mask_A[:X_row, X_row:]

    transposed_sub_matrix = np.transpose(sub_matrix)

    mask_A[X_row:, :transposed_sub_matrix.shape[1]] = transposed_sub_matrix
    Z = torch.matmul(mask_A, X_prime)
    train(args, Z, model_z, edge_index, edge_index1)
