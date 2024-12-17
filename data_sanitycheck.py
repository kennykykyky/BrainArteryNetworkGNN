import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from src.dataset import VascularDataset

# Helper function to set random seeds for reproducibility
def seed_everything(seed):
    print(f"Seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Helper function to summarize dataset
def summarize_dataset(dataset):
    print("\n--- Dataset Summary ---")
    print(f"Number of graphs: {len(dataset)}")
    
    node_counts = [data.num_nodes for data in dataset]
    edge_counts = [data.num_edges for data in dataset]
    edge_features_shape = [data.edge_attr.shape if hasattr(data, 'edge_attr') else None for data in dataset]
    node_features_shape = [data.x.shape if hasattr(data, 'x') else None for data in dataset]

    print(f"Node count: min={min(node_counts)}, max={max(node_counts)}, avg={np.mean(node_counts)}")
    print(f"Edge count: min={min(edge_counts)}, max={max(edge_counts)}, avg={np.mean(edge_counts)}")
    print(f"Edge features shapes: {set(edge_features_shape)}")
    print(f"Node features shapes: {set(node_features_shape)}")

    print(f"Example graph structure (first graph):\n")
    print(dataset[0])
    

# Sanity checks
def validate_dataset(dataset):
    print("\n--- Sanity Checks ---")
    # Check for NaNs in node features
    for i, data in enumerate(dataset):
        if hasattr(data, "x") and torch.isnan(data.x).any():
            print(f"Graph {i}: Node features contain NaNs.")

    # Check for NaNs in edge features
    for i, data in enumerate(dataset):
        if hasattr(data, "edge_attr") and torch.isnan(data.edge_attr).any():
            print(f"Graph {i}: Edge features contain NaNs.")

    # Check for zero or negative edge lengths if length is a feature
    for i, data in enumerate(dataset):
        if hasattr(data, "edge_attr"):
            lengths = data.edge_attr[:, 0]  # Assuming first column is length
            if (lengths <= 0).any():
                print(f"Graph {i}: Edge length <= 0 detected.")

    # # Check for disconnected graphs
    # for i, data in enumerate(dataset):
    #     if hasattr(data, "edge_index"):
    #         num_edges = data.edge_index.size(1)
    #         if num_edges < data.num_nodes - 1:  # A simple heuristic
    #             print(f"Graph {i}: Graph may be disconnected (num_edges < num_nodes - 1).")

# Plot 2D projection of node coordinates
def plot_node_projections(dataset):
    tmp_folder = "/home/kaiyu/research/git_repos/BrainGB/tmp/"
    for i, data in enumerate(dataset):
        if not hasattr(data, "pos"):
            print(f"Graph {i} does not have node positions. Skipping.")
            continue
        coords = data.pos.numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(coords[:, 0], coords[:, 1], c="blue", alpha=0.6, edgecolors="k")
        plt.title(f"Graph {i}: Node Coordinate Projection")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.savefig(os.path.join(tmp_folder, f"graph_{i}_projection.png"))
        plt.close()

# Plot statistics of node and edge features
def plot_feature_statistics(dataset):
    tmp_folder = "/home/kaiyu/research/git_repos/BrainGB/tmp/"
    node_means, node_stds, edge_means, edge_stds = [], [], [], []

    for data in dataset:
        if hasattr(data, "x"):
            node_means.append(data.x.mean(dim=0).numpy())
            node_stds.append(data.x.std(dim=0).numpy())
        if hasattr(data, "edge_attr"):
            edge_means.append(data.edge_attr.mean(dim=0).numpy())
            edge_stds.append(data.edge_attr.std(dim=0).numpy())

    if node_means:
        node_means = np.stack(node_means)
        plt.figure()
        plt.boxplot(node_means, vert=False)
        plt.title("Node Feature Means Across Graphs")
        plt.xlabel("Mean Value")
        plt.ylabel("Feature Dimension")
        plt.savefig(os.path.join(tmp_folder, "node_feature_means.png"))
        plt.close()

    if node_stds:
        node_stds = np.stack(node_stds)
        plt.figure()
        plt.boxplot(node_stds, vert=False)
        plt.title("Node Feature Standard Deviations Across Graphs")
        plt.xlabel("Standard Deviation")
        plt.ylabel("Feature Dimension")
        plt.savefig(os.path.join(tmp_folder, "node_feature_stds.png"))
        plt.close()

    if edge_means:
        edge_means = np.stack(edge_means)
        plt.figure()
        plt.boxplot(edge_means, vert=False)
        plt.title("Edge Feature Means Across Graphs")
        plt.xlabel("Mean Value")
        plt.ylabel("Feature Dimension")
        plt.savefig(os.path.join(tmp_folder, "edge_feature_means.png"))
        plt.close()

    if edge_stds:
        edge_stds = np.stack(edge_stds)
        plt.figure()
        plt.boxplot(edge_stds, vert=False)
        plt.title("Edge Feature Standard Deviations Across Graphs")
        plt.xlabel("Standard Deviation")
        plt.ylabel("Feature Dimension")
        plt.savefig(os.path.join(tmp_folder, "edge_feature_stds.png"))
        plt.close()


def main():
    seed = 42  # Set a seed for reproducibility
    seed_everything(seed)

    # Set dataset parameters
    dataset_name = "BrainVasculature"  # Change as needed
    node_features = "adj"  # Example: 'adj' or None for raw graphs
    self_dir = "/home/kaiyu/research/git_repos/BrainGB/examples"
    dataset_dir = os.path.join(self_dir, f'datasets/{dataset_name}/')

    # Construct the dataset
    dataset = VascularDataset(
        root=dataset_dir,
        name=dataset_name,
        pre_transform=None
    )

    # Summarize the dataset
    summarize_dataset(dataset)

    # Perform sanity checks
    validate_dataset(dataset)
    
    plot_node_projections(dataset)
    plot_feature_statistics(dataset)


if __name__ == "__main__":
    main()