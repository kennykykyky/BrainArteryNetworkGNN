import torch
import pdb
import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from torch_geometric.data import DataLoader

from src.dataset import VascularDataset
from src.models import GAT, GCN, BrainNN, MLP

def load_trained_model(model, checkpoint_path, device):
    """
    Load trained weights into the GAT model.

    Args:
        model: The GAT model instance.
        checkpoint_path: Path to the checkpoint file.

    Returns:
        model: The GAT model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Model weights loaded from {checkpoint_path}")
    return model

@torch.no_grad()
def calculate_node_edge_importance(model, data):
    """
    Calculate node and edge importance based on the trained GAT model.

    Args:
        model: The GAT model.
        data: PyG data object containing the graph.

    Returns:
        node_importance: Node importance values.
        edge_importance: Edge importance values.
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)  # Ensure data is on the same device as the model

    # Forward pass to compute attention scores
    node_importance = np.zeros(data.num_nodes)
    edge_importance_list = []  # To store edge importance for each layer

    model = model.gnn
    # Iterate over the convolutional layers
    for i, conv in enumerate(model.convs):
        # Extract the MPGATConv layer
        mpgat_conv = conv[0]

        # Perform the forward pass and store the attention scores (_alpha)
        x = mpgat_conv(data.x, data.edge_index, data.edge_attr)
        if hasattr(mpgat_conv, '_alpha') and mpgat_conv._alpha is not None:
            # Shape of _alpha: [num_edges, num_heads]
            edge_importance = mpgat_conv._alpha.detach().cpu().numpy()
            
            # Aggregate multi-head attention scores, e.g., mean across heads
            aggregated_edge_importance = edge_importance.mean(axis=1)  # [num_edges]
            edge_importance_list.append(aggregated_edge_importance)

            # Update node importance: Sum of attention scores for nodes
            for edge, attn in zip(data.edge_index.t().cpu().numpy(), aggregated_edge_importance):
                node_importance[edge[0]] += attn
                node_importance[edge[1]] += attn
        else:
            print(f"Warning: Attention coefficients not found in layer {i}")
        # data.x = x
        break # only one layer

    # Finalize results
    node_importance = node_importance
    edge_importance = edge_importance_list[0]

    return node_importance, edge_importance


import networkx as nx
import matplotlib.pyplot as plt

def visualize_with_matplotlib(data, node_importance, edge_importance, save_path="graph_visualization.png"):
    """
    Visualize node and edge importance using matplotlib.

    Args:
        data: PyG data object.
        node_importance: Node importance values.
        edge_importance: Edge importance values.
        save_path: Path to save the visualization.
    """
    G = nx.Graph()
    for node_id in range(data.num_nodes):
        G.add_node(node_id, pos=data.x[node_id, :2].cpu().numpy(), importance=node_importance[node_id])

    for (src, tgt), importance in zip(data.edge_index.t().cpu().numpy(), edge_importance):
        G.add_edge(src, tgt, weight=importance)

    pos = nx.get_node_attributes(G, 'pos')  # Get 2D positions
    node_colors = [G.nodes[n]['importance'] for n in G.nodes()]
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(12, 12))
    nx.draw(
        G, pos, node_size=300, node_color=node_colors, edge_color=edge_colors,
        cmap=plt.cm.Reds, edge_cmap=plt.cm.Blues, with_labels=True
    )
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), label="Node Importance")
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), label="Edge Importance")
    plt.title("Graph Visualization with Node and Edge Importance")
    plt.savefig(save_path)
    plt.show()
    
import plotly.graph_objects as go

def visualize_with_plotly(data, node_importance, edge_importance, save_path="graph_visualization.html"):
    """
    Visualize node and edge importance using Plotly in 3D with advanced rendering.

    Args:
        data: PyG data object.
        node_importance: Node importance values.
        edge_importance: Edge importance values.
    """
    edge_x, edge_y, edge_z = [], [], []
    for (src, tgt), importance in zip(data.edge_index.t().cpu().numpy(), edge_importance):
        x0, y0, z0 = data.x[src, :3].cpu().numpy()
        x1, y1, z1 = data.x[tgt, :3].cpu().numpy()
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color=edge_importance, colorscale='Blues', colorbar=dict(title="Edge Importance")),
        hoverinfo='none', mode='lines'
    )

    node_x, node_y, node_z, node_color = [], [], [], []
    for idx in range(data.num_nodes):
        x, y, z = data.x[idx, :3].cpu().numpy()
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_color.append(node_importance[idx])

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers',
        marker=dict(
            size=10 + 30 * (node_color / max(node_color)),  # Scale sizes
            color=node_color,
            colorscale='Reds',
            colorbar=dict(title="Node Importance"),
            symbol='circle'
        )
    )

    # Create spheres for nodes
    sphere_traces = []
    for idx in range(data.num_nodes):
        x, y, z = data.x[idx, :3].cpu().numpy()
        importance = node_importance[idx]
        sphere_trace = go.Mesh3d(
            x=[x], y=[y], z=[z],
            alphahull=0,
            opacity=0.6,
            color='red',
            intensity=[importance],
            colorscale='Reds',
            showscale=False
        )
        sphere_traces.append(sphere_trace)

    fig = go.Figure(data=[edge_trace, node_trace] + sphere_traces)
    fig.update_layout(title="3D Graph Visualization with Node and Edge Importance", showlegend=False)
    pio.write_html(fig, file=save_path, auto_open=False)
    # fig.show()

def build_model(args, device, model_name, num_features, num_nodes):
    """Builds the specified model based on the arguments."""
    if model_name == 'gcn':
        model = BrainNN(
            args,
            GCN(num_features, args, num_nodes, num_classes=2),
            MLP(2 * num_nodes, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2),
        ).to(device)
    elif model_name == 'gat':
        model = BrainNN(
            args,
            GAT(num_features, args, num_nodes, num_classes=2),
            MLP(2 * num_nodes, args.gat_hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2),
        ).to(device)
    else:
        raise ValueError(f"ERROR: Model variant \"{args.model_name}\" not found!")
    return model


def main():
    parser = argparse.ArgumentParser(description="GNN Model Training")

    # Dataset and Feature Arguments
    parser.add_argument('--dataset_name', type=str, default="BrainVasculature", help="Dataset name.")
    parser.add_argument('--view', type=int, default=1, help="Data view to use.")
    parser.add_argument('--node_features', type=str, 
                        choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 
                                 'diff_matrix', 'eigenvector', 'eigen_norm', 'adj_more'], 
                        default=None, help="Node features to use.")

    # Model Arguments
    parser.add_argument('--model_name', type=str, default='gat', help="Model type: 'gcn' or 'gat'.")
    parser.add_argument('--pooling', type=str, choices=['sum', 'concat', 'mean'], default='sum', help="Pooling type.")
    parser.add_argument('--gcn_mp_type', type=str, default="weighted_sum", help="GCN message passing type.")
    parser.add_argument('--gat_mp_type', type=str, default="attention_edge_weighted", help="GAT message passing type.")
    parser.add_argument('--n_GNN_layers', type=int, default=4, help="Number of GNN layers.")
    parser.add_argument('--n_MLP_layers', type=int, default=2, help="Number of MLP layers.")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads for GAT.")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden layer dimension.")
    parser.add_argument('--gat_hidden_dim', type=int, default=256, help="Hidden dimension for GAT.")
    parser.add_argument('--edge_emb_dim', type=int, default=256, help="Edge embedding dimension.")
    parser.add_argument('--edge_attr_size', type=int, default=2, help="Edge attribute size.")
    parser.add_argument('--dropout', type=float, default=0.8, help="Dropout rate.")

    # Training and Optimization
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer.")
    parser.add_argument('--mixup', type=int, default=0, choices=[0, 1], help="Enable mixup augmentation.")
    parser.add_argument('--train_batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('--test_batch_size', type=int, default=1, help="Testing batch size.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--test_interval', type=int, default=5, help="Interval for testing during training.")
    parser.add_argument('--k_fold_splits', type=int, default=10, help="Number of K-fold splits for cross-validation.")
    parser.add_argument('--repeat', type=int, default=1, help="Number of experiment repetitions.")

    # Miscellaneous
    parser.add_argument('--bucket_sz', type=float, default=0.05, help="Bucket size for processing.")
    parser.add_argument('--diff', type=float, default=0.2, help="Regularization parameter.")
    parser.add_argument('--seed', type=int, default=112078, help="Random seed for reproducibility.")
    parser.add_argument('--enable_nni', action='store_true', help="Enable NNI for hyperparameter tuning.")
    parser.add_argument('--device_number', type=int, default=0, help="GPU device number for training.")
    
    # === Configuration Parameters ===
    checkpoint_path = "/home/kaiyu/research/git_repos/BrainGB/exp/log/BrainVasculature_gat_None_sum_weighted_sum_attention_edge_weighted_8_256_256_256_0.001_0.005_0.8_101_0/best_acc_model.pth"  # Path to trained model checkpoint
    dataset_path = "/home/kaiyu/research/git_repos/BrainGB/examples/datasets/BrainVasculature"  # Path to dataset folder
    visualization_output_path = "./tmp"  # Folder to save visualizations
    
    use_plotly = True  # If True, use Plotly for visualization; otherwise, use Matplotlib

    # === Model and Dataset Initialization ===
    print("Initializing model and dataset...")
    args = parser.parse_args()
    
    num_features = 3  # Number of features per node
    num_nodes = 158  # Max number of nodes in the graph
    num_classes = 2  # Binary classification example
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the GAT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device, args.model_name, num_features, num_nodes)
    model = load_trained_model(model, checkpoint_path, device)

    # Load dataset
    print("Loading dataset...")
    dataset = VascularDataset(root=dataset_path,
                           name=args.dataset_name,
                           pre_transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Ensure model is on the same device as data
    model.to(device)

    # === Process Each Graph in the Dataset ===
    print("Processing graphs...")
    for i, data in enumerate(dataloader):
        print(f"Processing graph {i + 1}/{len(dataloader)}")
        data = data.to(device)

        # Calculate node and edge importance
        print("Calculating node and edge importance...")
        node_importance, edge_importance = calculate_node_edge_importance(model, data)

        # Visualize and save output
        print("Visualizing graph...")
        if use_plotly:
            save_path = f"{visualization_output_path}/graph_{i + 1}.html"
            visualize_with_plotly(data, node_importance, edge_importance, save_path)
        else:
            save_path = f"{visualization_output_path}/graph_{i + 1}.png"
            visualize_with_matplotlib(data, node_importance, edge_importance, save_path)
            print(f"Saved visualization to {save_path}")
        break

    print("Processing complete. Visualizations saved.")

# === Utility Functions (Ensure these are available/imported) ===
# load_trained_model
# calculate_node_edge_importance
# visualize_with_matplotlib
# visualize_with_plotly

if __name__ == "__main__":
    main()
