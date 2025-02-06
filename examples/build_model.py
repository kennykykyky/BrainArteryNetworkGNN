import torch
from src.models import GAT, GCN, BrainNN, MLP
from torch_geometric.data import Data
from typing import List


def build_model(args, device, model_name, num_features, num_nodes, n_classes=2):
    if model_name == 'gcn':
        model = BrainNN(args,
                      GCN(num_features, args, num_nodes, num_classes=n_classes),
                      MLP(2 * num_nodes, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=n_classes),
                      ).to(device)
    elif model_name == 'gat':
        model = BrainNN(args,
                      GAT(num_features, args, num_nodes, num_classes=n_classes),
                      MLP(2 * num_nodes, args.gat_hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=n_classes),
                      ).to(device)
    else:
        raise ValueError(f"ERROR: Model variant \"{args.variant}\" not found!")
    # check if model has regression
    if hasattr(args, 'regression') and args.regression:
        model.regression = True
    return model
