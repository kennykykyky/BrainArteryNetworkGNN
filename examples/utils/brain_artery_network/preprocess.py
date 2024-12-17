import os
import networkx as nx
import numpy as np
import scipy.io as sio
import pandas as pd
import pdb
import ast

def calculate_initial_features(G):
    # Calculate node degree centrality
    # degree_centrality = nx.degree_centrality(G)
    # nx.set_node_attributes(G, degree_centrality, 'degree_centrality')

    # Calculate edge length and set initial features (we'll calculate tortuosity later)
    for edge in G.edges(data=True):
        node1, node2 = edge[0], edge[1]
        
        # Calculate Euclidean distance (length) between nodes
        pos1 = np.array(ast.literal_eval(G.nodes[node1].get('networkx_key')).get('pos', None)[0])
        pos2 = np.array(ast.literal_eval(G.nodes[node2].get('networkx_key')).get('pos', None)[0])
        
        if np.isnan(pos1).any() or np.isnan(pos2).any():
            print(f"Warning: NaN found in node positions for edge ({node1}, {node2})")
        
        length = np.linalg.norm(pos1 - pos2)
        
        if np.isnan(length):
            print(f"Warning: NaN found in edge length for edge ({node1}, {node2})")
        
        # Store the actual segment length as `length` (to be summed up later for longer paths)
        G.edges[node1, node2]['length'] = length

    return G

def normalize_3d_positions(node_positions):
    """
    Normalize 3D positions based on mean and std of each subject's data.
    """
    mean_pos = node_positions.mean(axis=0, keepdims=True)
    std_pos = node_positions.std(axis=0, keepdims=True)
    normalized_positions = (node_positions - mean_pos) / (std_pos + 1e-6)  # Add epsilon to avoid division by zero
    return normalized_positions

def normalize_edge_features(edge_features):
    """
    Normalize edge features based on mean and std of each subject's data.
    """
    lengths = np.array([edge['length'] for edge in edge_features])
    tortuosities = np.array([edge['tortuosity'] for edge in edge_features])

    mean_length = lengths.mean()
    std_length = lengths.std()
    mean_tortuosity = tortuosities.mean()
    std_tortuosity = tortuosities.std()

    normalized_edge_features = []
    for edge in edge_features:
        normalized_edge = edge.copy()
        normalized_edge['length'] = (edge['length'] - mean_length) / (std_length + 1e-6)
        normalized_edge['tortuosity'] = (edge['tortuosity'] - mean_tortuosity) / (std_tortuosity + 1e-6)
        normalized_edge_features.append(normalized_edge)

    return normalized_edge_features

def simplify_graph_with_tortuosity(G):
    # Create a simplified graph
    simplified_G = nx.Graph()
    node_positions = []
    edge_features = []
    
    # Collect 3D positions of nodes to normalize
    for node in G.nodes:
        pos = np.array(ast.literal_eval(G.nodes[node].get('networkx_key')).get('pos', None)[0])
        node_positions.append(pos)

    # Convert to NumPy array for normalization
    node_positions = np.array(node_positions)

    # Normalize 3D positions per subject
    node_positions_normalized = normalize_3d_positions(node_positions)

    # Add leaf nodes and bifurcation nodes to the simplified graph
    for idx, node in enumerate(G.nodes):
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)

        if degree == 1 or degree >= 3 or (degree == 2 and len(set(G[node][nbr]['ves_type'] for nbr in neighbors)) > 1):
            pos = node_positions_normalized[idx]  # Use normalized position
            radius = ast.literal_eval(G.nodes[node].get('networkx_key')).get('radius', None)
            ves_type = ast.literal_eval(G.nodes[node].get('networkx_key')).get('ves_type', None)
            label = G.nodes[node].get('label')
            if np.isnan(pos).any():
                print(f"Warning: NaN found in 3D position of node {node}")
            if np.isnan(radius):
                print(f"Warning: NaN found in radius of node {node}")
            if np.isnan(ves_type).any():
                print(f"Warning: NaN found in ves_type of node {node}")
    
            simplified_G.add_node(node,
                                  pos=pos,
                                  radius=radius, 
                                  ves_type=ves_type, 
                                  label=label)
            
    # Iterate over simplified nodes and concatenate paths where necessary
    for node in list(simplified_G.nodes):
        for nbr in list(G.neighbors(node)):
            if nbr not in simplified_G:
                # Traverse and concatenate segments
                total_length, current_node, current_nbr = 0, node, nbr
                start_pos = np.array(ast.literal_eval(G.nodes[node].get('networkx_key')).get('pos', None)[0])
                end_pos = None
                combined_ves_type = set()
                
                if np.isnan(start_pos).any():
                    print(f"Warning: NaN found in start position of node {node}")
                
                while current_nbr not in simplified_G:
                    # Sum the actual segment length for concatenation
                    total_length += G[current_node][current_nbr]['length']
                    
                    combined_ves_type.add(G[current_node][current_nbr].get('ves_type', []))
                    
                    # Move to the next node along the path
                    next_nbrs = [n for n in G.neighbors(current_nbr) if n != current_node]
                    if not next_nbrs:
                        break
                    current_node, current_nbr = current_nbr, next_nbrs[0]
                
                # Calculate Euclidean distance between start and end nodes of concatenated path
                end_pos = np.array(ast.literal_eval(G.nodes[current_node].get('networkx_key')).get('pos', None)[0])
                euclidean_distance = np.linalg.norm(start_pos - end_pos)
                
                if np.isnan(end_pos).any():
                    print(f"Warning: NaN found in end position of node {current_node}")

                euclidean_distance = np.linalg.norm(start_pos - end_pos)

                if np.isnan(euclidean_distance):
                    print(f"Warning: NaN found in Euclidean distance for edge ({node}, {current_nbr})")
                
                # Calculate tortuosity as total length over Euclidean distance
                tortuosity = total_length / euclidean_distance if euclidean_distance > 0 else 1

                if np.isnan(tortuosity):
                    print(f"Warning: NaN found in tortuosity for edge ({node}, {current_nbr})")
                
                # Add the edge to the simplified graph
                if current_nbr in simplified_G:
                    simplified_G.add_edge(node, current_nbr, length=total_length, tortuosity=tortuosity, ves_type=list(combined_ves_type)[0])
                    edge_features.append({'length': total_length, 'tortuosity': tortuosity})

    # Normalize edge features
    length_mean, length_std = np.mean([edge['length'] for edge in edge_features]), np.std([edge['length'] for edge in edge_features])
    tortuosity_mean, tortuosity_std = np.mean([edge['tortuosity'] for edge in edge_features]), np.std([edge['tortuosity'] for edge in edge_features])
    edge_features = normalize_edge_features(edge_features)
    
    def normalize(value, mean, std):
        return (value - mean) / std
    
    # Update simplified graph with normalized edge features
    for edge in simplified_G.edges(data=True):
        node1, node2 = edge[0], edge[1]
        edge_data = edge[2]
        # Normalize edge_data
        normalized_length = normalize(edge_data['length'], length_mean, length_std)
        normalized_tortuosity = normalize(edge_data['tortuosity'], tortuosity_mean, tortuosity_std)
        for feature in edge_features:
            if np.isclose(normalized_length, feature['length']) and np.isclose(normalized_tortuosity, feature['tortuosity']):
                simplified_G.edges[node1, node2]['length'] = feature['length']
                simplified_G.edges[node1, node2]['tortuosity'] = feature['tortuosity']
                break
    
    return simplified_G

def load_gexf_graphs(data_folder, atlas_names):
    """
    Load GEXF files from a specified folder and return them as NetworkX graphs.
    """
    graph_data = {}
    for filename in os.listdir(data_folder):
        if filename.endswith(".gexf") and any(atlas_name in filename for atlas_name in atlas_names):
            subject_id = filename.split(".")[0]
            graph_path = os.path.join(data_folder, filename)
            G = nx.read_gexf(graph_path)
            G = calculate_initial_features(G)
            G = simplify_graph_with_tortuosity(G)
            graph_data[subject_id] = G
    return graph_data

def pad_matrix(matrices, target_size):
    padded_matrices = []
    for matrix in matrices:
        assert matrix.shape[0] == matrix.shape[1], "Connectivity matrix must be square."
        padded_matrix = np.zeros((target_size, target_size))
        current_size = matrix.shape[0]
        padded_matrix[:current_size, :current_size] = matrix
        padded_matrices.append(padded_matrix)
    return padded_matrices

def pad_node_positions(node_positions, target_size):
    padded_positions = []
    for positions in node_positions:
        padded_position = np.zeros((target_size, positions.shape[1]))
        current_size = positions.shape[0]
        padded_position[:current_size, :] = positions
        padded_positions.append(padded_position)
    return padded_positions

def save_data_as_npy(graph_data, atlas_name, metadata_file, save_path=None):
    """
    Extracts connectivity matrices from graphs, compiles metadata, and saves them in .npy format.
    Also saves edge features such as radius and length with information about the connected nodes.
    """
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    # id2site = metadata.set_index("subject")["SITE_ID"].to_dict()

    # Initialize data lists
    connectivity_matrices = []
    labels = []
    sites = []
    node_coordinates = []
    edge_features = []
    max_size = 0
    
    for subject_id, G in graph_data.items():
        
        # Extract connectivity matrix as binary adjacency matrix
        connectivity_matrix = nx.to_numpy_array(G)
        if "BRAVE" in subject_id:
            sub_id = 'BRAVE_' + subject_id.split("_")[-2]
        elif "CROP" in subject_id:
            sub_id = 'CROP_' + subject_id.split("_")[-3]
        elif "Dementia" in subject_id:
            sub_id = 'SKDementia_' + subject_id.split("_")[-3]
        
        # check whether sub_id is in metadata
        if sub_id not in metadata["ID"].values:
            print(f"Warning: {sub_id} not found in metadata")
            continue

        # Metadata for each subject
        # site = id2site.get(int(subject_id), None)  # Get the site ID
        # label = metadata.loc[metadata["ID"]==sub_id, "Gender"].values[0]
        # label = metadata.loc[metadata["ID"]==sub_id, "Diabetes"].values[0]
        label = metadata.loc[metadata["ID"]==sub_id, "Risk_category"].values[0]

        # Append data to respective lists
        connectivity_matrices.append(connectivity_matrix)
        if connectivity_matrix.shape[0] > max_size:
            max_size = connectivity_matrix.shape[0]
        labels.append(label)
        # sites.append(site)

        # Extract edge features (radius, length, node connections)
        subject_edge_features = []
        for node1, node2, edge_data in G.edges(data=True):
            node_idx1 = list(G.nodes).index(node1)
            node_idx2 = list(G.nodes).index(node2)
            radius = edge_data.get('radius', None)
            length = edge_data.get('length', None)
            tortuosity = edge_data.get('tortuosity', None)
            
            # Store edge feature as a dictionary with connected nodes and attributes
            edge_feature = {
                "node1": node_idx1,
                "node2": node_idx2,
                "radius": radius,
                "length": length,
                "tortuosity": tortuosity
            }
            subject_edge_features.append(edge_feature)
        
        edge_features.append(subject_edge_features)

        # Extract node coordinates
        node_coords = []
        for node in G.nodes(data=True):
            pos_str = node[1]
            if pos_str is not None:
                pos = pos_str.get('pos')  # Default to [0,0,0] if not found
            else:
                print(f"Warning: 'pos' not found for node {node[0]}")
                pos = [0, 0, 0]  # Default value if 'pos' is not available
            node_coords.append(pos)
        
        node_coordinates.append(np.array(node_coords))

    # Convert lists to numpy arrays
    # connectivity_matrices = np.array(connectivity_matrices)
    connectivity_matrices = np.array(pad_matrix(connectivity_matrices, max_size))
    labels = np.array(labels)
    # sites = np.array(sites)
    # node_coordinates = np.array(node_coordinates)
    node_coordinates = np.array(pad_node_positions(node_coordinates, max_size))

    # Save to .npy format
    if save_path is None:
        save_path = os.getcwd()
    
    np.save(os.path.join(save_path, f'{atlas_name}.npy'), {
        "connectivity": connectivity_matrices,
        "label": labels,
        # "site": sites,
        "node_coordinates": node_coordinates,
        "edge_features": edge_features
    })

    print(f"Data saved in {save_path}/{atlas_name}.npy")

# Path to your folder containing the GEXF files
data_folder = '/home/kaiyu/research/VesselSeg/data/multiclass_lumen/graphs'
metadata_file = '/home/kaiyu/research/git_repos/BrainGB/examples/datasets/BrainVasculature/Formal_CROP-BRAVE-IPH-SKDementia_DemoClin.csv'  # Path to your metadata CSV file

# can generate different subset of data
atlas_names = ['BRAVE', 'CROP', 'Dementia']  # Specify the dataset or atlas name
# dataset_name = "BrainVasculature"  # Specify the dataset name
dataset_name = "BrainVasculature_Framingham"  # Specify the dataset name
# save_path = '/home/kaiyu/research/git_repos/BrainGB/examples/datasets/BrainVasculature'
save_path = '/home/kaiyu/research/git_repos/BrainGB/examples/datasets/BrainVasculature_Framingham'

os.makedirs(save_path, exist_ok=True)

# Load GEXF files
graph_data = load_gexf_graphs(data_folder, atlas_names)
# Save the data in .npy format
save_data_as_npy(graph_data, dataset_name, metadata_file, save_path)
