
import torch_geometric.transforms as T
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import WikiCS
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os

def DataLoader(name):
    name = name.lower()
    root_path = './data/'
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root_path, name, split='random', num_train_per_class=20, num_val=500, num_test=1000,
                            transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        dataset = Amazon(root_path, name, T.NormalizeFeatures())

    elif name in ['cs', 'physics']:
        dataset = Coauthor(root_path, name, T.NormalizeFeatures())

    elif name in ['chameleon', 'squirrel']:
        preProcDs = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        dataset.data = data
        return dataset

    elif name in ['film']:
        dataset = Actor(root=root_path + '/Actor', transform=T.NormalizeFeatures())
        dataset.name = name
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root=root_path, name=name, transform=T.NormalizeFeatures())
    elif name in ['wikics']:
        dataset = WikiCS(root=root_path + '/WikiCS', transform=T.NormalizeFeatures())
    elif name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(root=root_path,
                                         name='ogbn-arxiv')
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    return dataset

