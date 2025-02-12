import os
import networkx as nx
import numpy as np
import scipy.io as sio
import pandas as pd
import pdb
import ast
import pickle

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
        r1 = ast.literal_eval(G.nodes[node1].get('networkx_key')).get('radius', None)
        r2 = ast.literal_eval(G.nodes[node2].get('networkx_key')).get('radius', None)
        if r1 is not None and r2 is not None:
            radius = (r1 + r2)  # Alternatively, use (r1 + r2)/2 or 2*r if you prefer
        elif r1 is not None:
            radius = 2 * r1
        elif r2 is not None:
            radius = 2 * r2
        
        if np.isnan(length):
            print(f"Warning: NaN found in edge length for edge ({node1}, {node2})")
        
        if np.isnan(radius):
            print(f"Warning: NaN found in edge radius for edge ({node1}, {node2})")
        # Store the actual segment length as `length` (to be summed up later for longer paths)
        G.edges[node1, node2]['length'] = length
        G.edges[node1, node2]['radius'] = radius
        G.edges[node1, node2]['ves_type'] = edge[2].get('ves_type')
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

# def simplify_graph_with_tortuosity(G):
#     # Create a simplified graph
#     simplified_G = nx.Graph()
#     node_positions = []
#     edge_features = []
    
#     # Collect 3D positions of nodes to normalize
#     for node in G.nodes:
#         pos = np.array(ast.literal_eval(G.nodes[node].get('networkx_key')).get('pos', None)[0])
#         node_positions.append(pos)

#     # Convert to NumPy array for normalization
#     node_positions = np.array(node_positions)

#     # Normalize 3D positions per subject
#     node_positions_normalized = normalize_3d_positions(node_positions)

#     # Add leaf nodes and bifurcation nodes to the simplified graph
#     for idx, node in enumerate(G.nodes):
#         neighbors = list(G.neighbors(node))
#         degree = len(neighbors)

#         if degree == 1 or degree >= 3 or (degree == 2 and len(set(G[node][nbr]['ves_type'] for nbr in neighbors)) > 1):
#             pos = node_positions_normalized[idx]  # Use normalized position
#             radius = ast.literal_eval(G.nodes[node].get('networkx_key')).get('radius', None)
#             ves_type = ast.literal_eval(G.nodes[node].get('networkx_key')).get('ves_type', None)
#             label = G.nodes[node].get('label')
#             if np.isnan(pos).any():
#                 print(f"Warning: NaN found in 3D position of node {node}")
#             if np.isnan(radius):
#                 print(f"Warning: NaN found in radius of node {node}")
#             if np.isnan(ves_type).any():
#                 print(f"Warning: NaN found in ves_type of node {node}")
    
#             simplified_G.add_node(node,
#                                   pos=pos,
#                                   radius=radius, 
#                                   ves_type=ves_type, 
#                                   label=label)
            
#     # Iterate over simplified nodes and concatenate paths where necessary
#     for node in list(simplified_G.nodes):
#         for nbr in list(G.neighbors(node)):
#             if nbr not in simplified_G:
#                 # Traverse and concatenate segments
#                 total_length, current_node, current_nbr = 0, node, nbr
#                 start_pos = np.array(ast.literal_eval(G.nodes[node].get('networkx_key')).get('pos', None)[0])
#                 end_pos = None
#                 combined_ves_type = set()
                
#                 if np.isnan(start_pos).any():
#                     print(f"Warning: NaN found in start position of node {node}")
                
#                 while current_nbr not in simplified_G:
#                     # Sum the actual segment length for concatenation
#                     total_length += G[current_node][current_nbr]['length']
                    
#                     combined_ves_type.add(G[current_node][current_nbr].get('ves_type', []))
                    
#                     # Move to the next node along the path
#                     next_nbrs = [n for n in G.neighbors(current_nbr) if n != current_node]
#                     if not next_nbrs:
#                         break
#                     current_node, current_nbr = current_nbr, next_nbrs[0]
                
#                 # Calculate Euclidean distance between start and end nodes of concatenated path
#                 end_pos = np.array(ast.literal_eval(G.nodes[current_node].get('networkx_key')).get('pos', None)[0])
#                 euclidean_distance = np.linalg.norm(start_pos - end_pos)
                
#                 if np.isnan(end_pos).any():
#                     print(f"Warning: NaN found in end position of node {current_node}")

#                 euclidean_distance = np.linalg.norm(start_pos - end_pos)

#                 if np.isnan(euclidean_distance):
#                     print(f"Warning: NaN found in Euclidean distance for edge ({node}, {current_nbr})")
                
#                 # Calculate tortuosity as total length over Euclidean distance
#                 tortuosity = total_length / euclidean_distance if euclidean_distance > 0 else 1

#                 if np.isnan(tortuosity):
#                     print(f"Warning: NaN found in tortuosity for edge ({node}, {current_nbr})")
                
#                 # Add the edge to the simplified graph
#                 if current_nbr in simplified_G:
#                     simplified_G.add_edge(node, current_nbr, length=total_length, tortuosity=tortuosity, ves_type=list(combined_ves_type)[0])
#                     edge_features.append({'length': total_length, 'tortuosity': tortuosity})

#     # Normalize edge features
#     length_mean, length_std = np.mean([edge['length'] for edge in edge_features]), np.std([edge['length'] for edge in edge_features])
#     tortuosity_mean, tortuosity_std = np.mean([edge['tortuosity'] for edge in edge_features]), np.std([edge['tortuosity'] for edge in edge_features])
#     edge_features = normalize_edge_features(edge_features)
    
#     def normalize(value, mean, std):
#         return (value - mean) / std
    
#     # Update simplified graph with normalized edge features
#     for edge in simplified_G.edges(data=True):
#         node1, node2 = edge[0], edge[1]
#         edge_data = edge[2]
#         # Normalize edge_data
#         normalized_length = normalize(edge_data['length'], length_mean, length_std)
#         normalized_tortuosity = normalize(edge_data['tortuosity'], tortuosity_mean, tortuosity_std)
#         for feature in edge_features:
#             if np.isclose(normalized_length, feature['length']) and np.isclose(normalized_tortuosity, feature['tortuosity']):
#                 simplified_G.edges[node1, node2]['length'] = feature['length']
#                 simplified_G.edges[node1, node2]['tortuosity'] = feature['tortuosity']
#                 break
    
#     return simplified_G

def normalize_edge_features_in_graph(G):
    """
    Normalize edge features (length, tortuosity, diameter) for all edges in graph G.
    """
    lengths = []
    tortuosities = []
    diameters = []
    for u, v, data in G.edges(data=True):
        lengths.append(data.get('length', 0))
        tortuosities.append(data.get('tortuosity', 1))
        if data.get('diameter', None) is not None:
            diameters.append(data.get('diameter'))
    
    # Normalize length and tortuosity
    lengths = np.array(lengths)
    tortuosities = np.array(tortuosities)
    if len(lengths) > 0:
        mean_length = lengths.mean()
        std_length = lengths.std() + 1e-6
        mean_tort = tortuosities.mean()
        std_tort = tortuosities.std() + 1e-6
    else:
        mean_length = std_length = mean_tort = std_tort = 1.0

    for u, v, data in G.edges(data=True):
        data['length'] = (data['length'] - mean_length) / std_length
        data['tortuosity'] = (data['tortuosity'] - mean_tort) / std_tort
        # Optionally, normalize diameter if needed:
        if data.get('diameter', None) is not None:
            # Here, we normalize diameter similarly.
            # You can adjust or even skip normalization for diameter if it is desired to keep in original scale.
            data['diameter'] = (data['diameter'] - np.mean(diameters)) / (np.std(diameters) + 1e-6)
    return G

def assign_segment_tortuosity(G):
    """
    For arterial segments that are composed of multiple edges (i.e. a chain of edges where 
    intermediate nodes have degree 2), compute the tortuosity for the entire segment and 
    assign it to all edges in that segment.
    
    The process:
      - Identify endpoints of segments: nodes with degree != 2.
      - For each such node, for each neighbor that has not been processed,
        follow the chain until reaching another node with degree != 2.
      - Compute the segment's actual length as the sum of the edge lengths.
      - Compute the chord length as the Euclidean distance between the first and last node.
      - If chord length > 0, set segment tortuosity = (actual length / chord length); else, use 1.
      - Assign this tortuosity value to all edges along that segment.
    """
    processed_edges = set()
    for node in list(G.nodes):
        # Process only if node is a bifurcation or endpoint (degree != 2)
        if G.degree(node) != 2:
            for neighbor in G.neighbors(node):
                edge_key = tuple(sorted([node, neighbor]))
                if edge_key in processed_edges:
                    continue
                # Begin a chain from the current node
                chain_nodes = [node, neighbor]
                prev = node
                current = neighbor
                # Follow the chain while current node is intermediate (degree == 2)
                while G.degree(current) == 2:
                    next_nodes = [n for n in G.neighbors(current) if n != prev]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0]
                    chain_nodes.append(next_node)
                    prev, current = current, next_node
                # Compute the actual length (sum of lengths of edges along the chain)
                actual_length = 0
                for i in range(len(chain_nodes) - 1):
                    actual_length += G.edges[chain_nodes[i], chain_nodes[i+1]]['length']
                # Compute chord length: Euclidean distance between first and last node positions
                # check virtual node
                if 'virtual' == chain_nodes[0] or 'virtual' == chain_nodes[-1]:
                    continue
                pos_start = np.array(G.nodes[chain_nodes[0]]['raw_pos'])
                pos_end = np.array(G.nodes[chain_nodes[-1]]['raw_pos'])
                chord_length = np.linalg.norm(pos_start - pos_end)
                segment_tortuosity = actual_length / chord_length if chord_length > 0 else actual_length
                # Assign the segment tortuosity to all edges in the chain
                for i in range(len(chain_nodes) - 1):
                    G.edges[chain_nodes[i], chain_nodes[i+1]]['tortuosity'] = segment_tortuosity
                    processed_edges.add(tuple(sorted([chain_nodes[i], chain_nodes[i+1]])))
    return G

def process_graph_without_subsampling(G):
    """
    Process the input graph G without subsampling nodes.
    Steps:
      1. Normalize 3D positions of nodes per subject.
      2. Create a new graph with all nodes (with normalized positions and additional attributes).
      3. Add edges from G with computed features:
         - length: Euclidean distance between normalized node positions.
         - diameter: computed as the sum (or average) of the radii from the two connected nodes.
         - tortuosity: if not provided, set to a default value of 1.
         - ves_type (artery label): extracted from node attributes.
      4. Check for connectivity and, if the graph is not connected, add a virtual node that connects to one representative node from each component.
    """
    processed_G = nx.Graph()
    node_list = list(G.nodes)
    raw_positions = []

    # Gather raw positions from nodes
    for node in node_list:
        node_attr = ast.literal_eval(G.nodes[node].get('networkx_key'))
        pos = np.array(node_attr.get('pos', None)[0])
        raw_positions.append(pos)
    raw_positions = np.array(raw_positions)
    normalized_positions = normalize_3d_positions(raw_positions)

    # Add all nodes with normalized positions and attributes
    for idx, node in enumerate(node_list):
        node_attr = ast.literal_eval(G.nodes[node].get('networkx_key'))
        pos_norm = normalized_positions[idx]
        pos = raw_positions[idx]
        radius = node_attr.get('radius', None)
        ves_type = node_attr.get('ves_type', None)  # Artery label
        label = G.nodes[node].get('label', None)
        processed_G.add_node(node, pos=pos_norm, raw_pos=pos, radius=radius, ves_type=ves_type, label=label)

    # Add edges with computed features
    for edge in G.edges(data=True):
        node1, node2 = edge[0], edge[1]
        
        diameter = edge[2].get('radius', None)
        length = edge[2].get('length', None)

        # Use provided tortuosity if available; else set default value of 1
        tortuosity = edge[2].get('tortuosity', 1)

        # For ves_type (artery label), use edge attribute if present, otherwise use node1's ves_type
        ves_type_edge = edge[2].get('ves_type', processed_G.nodes[node1].get('ves_type'))
        
        processed_G.add_edge(node1, node2,
                             length=length,
                             diameter=diameter,
                             tortuosity=tortuosity,
                             ves_type=ves_type_edge)
    
    # Virtual Connectivity: Check if graph is connected; if not, add a virtual node.
    if not nx.is_connected(processed_G):
        virtual_node = 'virtual'
        processed_G.add_node(virtual_node, pos=np.zeros(3), raw_pos=np.zeros(3), radius=None, ves_type='virtual', label='virtual')
        for comp in nx.connected_components(processed_G):
            if virtual_node in comp:
                continue
            # Choose a representative node from the component
            rep_node = list(comp)[0]
            processed_G.add_edge(virtual_node, rep_node,
                                 length=0,
                                 diameter=0,
                                 tortuosity=1,
                                 ves_type='virtual')
    processed_G = assign_segment_tortuosity(processed_G)
    processed_G = normalize_edge_features_in_graph(processed_G)
    
    return processed_G

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
            # G = simplify_graph_with_tortuosity(G)
            G = process_graph_without_subsampling(G)
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

def get_unique_vessel_types(graph_data):
    """
    Collect all unique vessel types from the graph data.
    Excludes 'virtual' vessel type and None values.
    """
    vessel_types = set()
    for G in graph_data.values():
        for _, _, edge_data in G.edges(data=True):
            ves_type = edge_data.get('ves_type', None)
            if ves_type is not None and ves_type != "virtual":  # Changed from 'or' to 'and'
                vessel_types.add(ves_type)
    return sorted(list(vessel_types))  # Sort to ensure consistent encoding

def encode_vessel_type(ves_type, vessel_types):
    """
    Create one-hot encoding for vessel type.
    """
    encoding = np.zeros(len(vessel_types))
    if ves_type is not None:
        try:
            idx = vessel_types.index(ves_type)
            encoding[idx] = 1
        except ValueError:
            pass  # Keep zero vector for unknown vessel types
    return encoding

def save_data_as_npy(graph_data, atlas_name, metadata_file, save_path=None):
    # Load metadata
    metadata = pd.read_csv(metadata_file)

    # Get unique vessel types for one-hot encoding
    vessel_types = get_unique_vessel_types(graph_data)
    print(f"Found {len(vessel_types)} unique vessel types: {vessel_types}")

    # Initialize data lists
    connectivity_matrices = []
    clinical_variables = {
        'gender': [],           # Binary
        'age': [],             # Continuous
        'total_cholesterol': [],# Continuous (TC)
        'ldl': [],             # Continuous
        'hdl': [],             # Continuous
        'sbp': [],             # Continuous
        'diabetes': [],         # Binary
        'smoking': [],         # Binary
        'framingham_risk': [], # Continuous (target score)
        'hypertension': [],    # Binary
        'risk_category': []    # Categorical
    }
    node_coordinates = []
    edge_features = []
    max_size = 0
    
    for subject_id, G in graph_data.items():
        connectivity_matrix = nx.to_numpy_array(G)
        if "BRAVE" in subject_id:
            sub_id = 'BRAVE_' + subject_id.split("_")[-2]
        elif "CROP" in subject_id:
            sub_id = 'CROP_' + subject_id.split("_")[-3]
        elif "Dementia" in subject_id:
            sub_id = 'SKDementia_' + subject_id.split("_")[-3]
        
        if sub_id not in metadata["ID"].values:
            print(f"Warning: {sub_id} not found in metadata")
            continue

        # Extract all clinical variables from metadata
        subject_data = metadata.loc[metadata["ID"] == sub_id]
        clinical_variables['gender'].append(subject_data["Gender"].values[0])
        clinical_variables['age'].append(subject_data["Age"].values[0])
        clinical_variables['total_cholesterol'].append(subject_data["TC"].values[0])
        clinical_variables['ldl'].append(subject_data["LDL"].values[0])
        clinical_variables['hdl'].append(subject_data["HDL"].values[0])
        clinical_variables['sbp'].append(subject_data["SBP"].values[0])
        clinical_variables['diabetes'].append(subject_data["Diabetes"].values[0])
        clinical_variables['smoking'].append(subject_data["Smoking"].values[0])
        clinical_variables['framingham_risk'].append(subject_data["Framingham_Risk"].values[0] / 100)
        clinical_variables['hypertension'].append(subject_data["Hypertension"].values[0])
        clinical_variables['risk_category'].append(subject_data["Risk_category"].values[0])

        connectivity_matrices.append(connectivity_matrix)
        if connectivity_matrix.shape[0] > max_size:
            max_size = connectivity_matrix.shape[0]

        # Extract edge features
        subject_edge_features = []
        for node1, node2, edge_data in G.edges(data=True):
            node_idx1 = list(G.nodes).index(node1)
            node_idx2 = list(G.nodes).index(node2)
            radius = edge_data.get('diameter', None)
            length = edge_data.get('length', None)
            tortuosity = edge_data.get('tortuosity', None)
            ves_type = edge_data.get('ves_type', None)
            
            # Create one-hot encoding for vessel type
            ves_type_encoding = encode_vessel_type(ves_type, vessel_types)
            
            edge_feature = {
                "node1": node_idx1,
                "node2": node_idx2,
                "diameter": radius,
                "length": length,
                "tortuosity": tortuosity,
                "ves_type": ves_type_encoding.tolist()  # Store the one-hot encoding
            }
            subject_edge_features.append(edge_feature)
        edge_features.append(subject_edge_features)

        # Extract node coordinates
        node_coords = []
        for node in G.nodes(data=True):
            pos = node[1].get('pos', [0, 0, 0])
            node_coords.append(pos)
        node_coordinates.append(np.array(node_coords))

    # Convert all data to numpy arrays
    connectivity_matrices = np.array(pad_matrix(connectivity_matrices, max_size))
    node_coordinates = np.array(pad_node_positions(node_coordinates, max_size))
    
    # Convert clinical variables to numpy arrays
    for key in clinical_variables:
        clinical_variables[key] = np.array(clinical_variables[key])

    if save_path is None:
        save_path = os.getcwd()
    
    # Save all data with protocol 5 for large file support
    save_dict = {
        "connectivity": connectivity_matrices,
        "node_coordinates": node_coordinates,
        "edge_features": edge_features,
        "clinical_variables": clinical_variables,
        "vessel_types": vessel_types  # Save the vessel types list for reference
    }
    
    output_path = os.path.join(save_path, f'{atlas_name}.npy')
    
    # Use pickle directly for saving large data
    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f, protocol=5)

    print(f"Data saved in {output_path}")

# Path configurations
data_folder = '/home/kaiyu/research/VesselSeg/data/multiclass_lumen/graphs'
metadata_file = '/home/kaiyu/research/git_repos/BrainArteryNetworkGNN/examples/datasets/Formal_CROP-BRAVE-IPH-SKDementia_DemoClin.csv'
atlas_names = ['BRAVE', 'CROP', 'Dementia']

# Base save path
base_save_path = '/home/kaiyu/research/git_repos/BrainArteryNetworkGNN/examples/datasets/BrainVasculature'

# Create different datasets for each target variable
target_variables = [
    'Gender',
    'Age',
    'TC',
    'LDL',
    'HDL',
    'SBP',
    'Diabetes',
    'Smoking',
    'FraminghamScore',
    'Hypertension',
    'RiskCategory'
]

# Load GEXF files once
graph_data = load_gexf_graphs(data_folder, atlas_names)

# Create datasets for each target variable
for target in target_variables:
    dataset_name = f"BrainVasculature_{target}"
    save_path = os.path.join(base_save_path, target)
    os.makedirs(save_path, exist_ok=True)
    
    # Save the data in .npy format
    save_data_as_npy(graph_data, dataset_name, metadata_file, save_path)
