import os
import torch
from tqdm import tqdm
import argparse
from torch.optim import Adam, AdamW, Adamax
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data
from GCL.eval import get_split
from Model import Model
from data_utils import DataLoader
from utils import get_ppmi_matrix, get_edge_index
from mask import MaskEdge, MaskPath
from eval import LREvaluator

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run(model, opt, args):
    model.train()
    with tqdm(total=args.epochs, desc='(Train)') as pbar:
        for i in range(1, args.epochs + 1):
            opt.zero_grad()
            loss = model(new_data, adj_dense_matrix)
            opt.step()
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()


def test(model, args):
    model.eval()
    z = model.get_emb(new_data, adj_dense_matrix)
    z1 = z[:num_nodes]
    z2 = z[num_nodes:]
    z2 = torch.matmul(x, z2)
    z = torch.cat([z1, z2], dim=1)
    split = get_split(num_samples=z.size(0), train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, label, split)
    return result


def main(args):
    activation = ({'relu': nn.ReLU, 'prelu': nn.PReLU, 'lrelu': nn.LeakyReLU, 'elu': nn.ELU})[args.act_fun]
    model = Model(num_nodes, feat_dim, args.reconstruct_dim, args.hidden_dim, args.decoder_layers, activation,
                  args.dropout, mask, ).to(device)

    opt = Adamax([
        dict(params=model.params_1, weight_decay=args.weight_decay1, lr=args.lr1),
        dict(params=model.params_2, weight_decay=args.weight_decay2, lr=args.lr2)
    ])

    for i in range(args.runs):
        run(model, opt, args)
        test(model, args)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--act_fun', type=str, default='elu')
    parser.add_argument('--lr1', type=float, default=0.001)
    parser.add_argument('--lr2', type=float, default=0.0005)
    parser.add_argument('--weight_decay1', type=float, default=1e-5)
    parser.add_argument('--weight_decay2', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--mask_type', type=str, default='mask_edge')
    parser.add_argument('--mask_rate', type=float, default=0.5)
    parser.add_argument('--reconstruct_dim', type=int, default=2048)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--decoder_layers', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(f'Using device {device}')
    dataset = DataLoader(args.dataset)
    data = dataset[0]
    label = data.y
    x = data.x
    edge_index = data.edge_index
    num_nodes = x.shape[0]
    feat_dim = x.shape[1]
    matrix = to_scipy_sparse_matrix(edge_index).todense()
    matrix = torch.tensor(matrix, dtype=torch.float)
    T = get_ppmi_matrix(x, args.dataset)
    edge_index_new, adj_dense_matrix = get_edge_index(x, matrix)
    if args.mask_type == 'mask_edge':
        mask = MaskEdge(p=args.mask_rate)
    new_data = Data(orgin_x=x, edge_index=edge_index_new, origin_edge_index=edge_index, matrix=matrix, T=T,
                    name=args.dataset, adj_dense_matrix=adj_dense_matrix)
    new_data = new_data.to(device)
    adj_dense_matrix = adj_dense_matrix.to(device)
    data = data.to(device)
    x = x.to(device)
    main(args)
