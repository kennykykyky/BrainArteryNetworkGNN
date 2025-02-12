import torch
from src.models import GAT, GCN, BrainNN, MLP, GraphTransformer
from torch_geometric.data import Data
from typing import List


def build_model(args, device, model_name, num_features, num_nodes, num_edge_features=None, n_classes=2):
    """
    Build the model with the specified architecture.
    
    Args:
        args: Arguments containing model parameters
        device: Device to run the model on
        model_name: Name of the model architecture
        num_features: Number of node features
        num_nodes: Number of nodes in the graph
        num_edge_features: Number of edge features (including vessel type encoding)
        n_classes: Number of output classes (2 for binary classification, 1 for regression)
    """
    if model_name == 'gcn':
        model = BrainNN(args,
                      GCN(num_features, args, num_nodes, num_edge_features=num_edge_features, num_classes=n_classes),
                      None,
                    #   MLP(2 * num_nodes, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=n_classes),
                      ).to(device)
    elif model_name == 'gat':
        model = BrainNN(args,
                      GAT(num_features, args, num_nodes, num_edge_features=num_edge_features, num_classes=n_classes),
                      None,
                    #   MLP(2 * num_nodes, args.gat_hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=n_classes),
                      ).to(device)
    elif args.model_name == "gps":
        model = BrainNN(args,
                      GraphTransformer(num_features, args, num_nodes, num_edge_features=num_edge_features, num_classes=n_classes),
                      None,
                      ).to(device)
    else:
        raise ValueError(f"ERROR: Model variant \"{args.variant}\" not found!")
    
    # Set regression mode if specified
    if hasattr(args, 'regression') and args.regression:
        model.regression = True
    return model
