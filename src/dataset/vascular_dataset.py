import torch
import os
import numpy as np
import os.path as osp

from scipy.io import loadmat
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
from .base_transform import BaseTransform
from torch_geometric.utils import degree
import sys
from torch_geometric.data.makedirs import makedirs
from .abcd.load_abcd import load_data
from torch_geometric.data.dataset import files_exist
import logging
import pdb

def dense_to_ind_val(adj, length_matrix=None, tortuosity_matrix=None, radius_matrix=None):
    """
    Converts dense adjacency and feature matrices to sparse edge representation.
    
    Parameters:
    - adj: torch.Tensor, the adjacency matrix.
    - length_matrix: torch.Tensor, the matrix representing edge lengths.
    - tortuosity_matrix: torch.Tensor, the matrix representing edge tortuosities.
    - radius_matrix: torch.Tensor, the matrix representing edge radii.
    
    Returns:
    - edge_index: torch.Tensor, indices of connected nodes (2, num_edges).
    - edge_attr: torch.Tensor, edge attributes (num_edges, num_features).
    """
    
    # convert all inputs to torch Tensors
    adj = torch.as_tensor(adj)
    if length_matrix is not None:
        length_matrix = torch.as_tensor(length_matrix)
    if tortuosity_matrix is not None:
        tortuosity_matrix = torch.as_tensor(tortuosity_matrix)
    if radius_matrix is not None:
        radius_matrix = torch.as_tensor(radius_matrix)
    
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)
    
    # Identify non-zero (or non-NaN) entries in the adjacency matrix
    index = (torch.isnan(adj) == 0).nonzero(as_tuple=True)
    edge_attr = adj[index].unsqueeze(1)  # Adjacency values as edge features (make it (num_edges, 1))
    
    # Initialize a list to hold all edge attributes
    edge_features = [edge_attr]
    
    # Add length features if provided
    if length_matrix is not None:
        length_features = length_matrix[index].unsqueeze(1)
        edge_features.append(length_features)
    
    # Add tortuosity features if provided
    if tortuosity_matrix is not None:
        tortuosity_features = tortuosity_matrix[index].unsqueeze(1)
        edge_features.append(tortuosity_features)
    
    # Add radius features if provided
    if radius_matrix is not None:
        radius_features = radius_matrix[index].unsqueeze(1)
        edge_features.append(radius_features)
    
    # Concatenate all edge features along the last dimension
    edge_attr = torch.cat(edge_features, dim=1)
    
    return torch.stack(index, dim=0), edge_attr

class VascularDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform: BaseTransform = None, view=0):
        self.view: int = view
        self.name = name.upper()
        self.filename_postfix = str(pre_transform) if pre_transform is not None else None
        # assert self.name in ['BRAVE', 'SKDementia', 'CROP', 'IPH', 'BRAINVASCULATURE']
        super(VascularDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.num_nodes = torch.load(self.processed_paths[0])
        self.pre_transform = pre_transform
        logging.info('Loaded dataset: {}'.format(self.name))

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.mat'

    @property
    def processed_file_names(self):
        name = f'{self.name}_{self.view}'
        if self.filename_postfix is not None:
            name += f'_{self.filename_postfix}'
        return f'{name}.pt'

    # def download(self):
    #     raise NotImplementedError

    # def process(self):
    #     if self.name in ['BRAVE', 'SKDementia', 'CROP', 'IPH', 'BRAINVASCULATURE']:
    #         adj, edge_features, node_features, y = load_data(self.raw_dir)
    #         y = torch.LongTensor(y)
    #         adj = torch.Tensor(adj)
    #         num_graphs = adj.shape[0]
    #         num_nodes = adj.shape[1]
    #     else:
    #         m = loadmat(osp.join(self.raw_dir, self.raw_file_names))
    #         if self.name == 'PPMI':
    #             if self.view > 2 or self.view < 0:
    #                 raise ValueError(f'{self.name} only has 3 views')
    #             raw_data = m['X']
    #             num_graphs = raw_data.shape[0]
    #             num_nodes = raw_data[0][0].shape[0]
    #             a = np.zeros((num_graphs, num_nodes, num_nodes))
    #             for i, sample in enumerate(raw_data):
    #                 a[i, :, :] = sample[0][:, :, self.view]
    #             adj = torch.Tensor(a)
    #         else:
    #             key = 'fmri' if self.view == 1 else 'dti'
    #             adj = torch.Tensor(m[key]).transpose(0, 2)
    #             num_graphs = adj.shape[0]
    #             num_nodes = adj.shape[1]

    #         y = torch.Tensor(m['label']).long().flatten()
    #         y[y == -1] = 0

    #     data_list = []
    #     for i in range(num_graphs):
    #         # Initialize edge feature matrices
    #         length_matrix = np.zeros((num_nodes, num_nodes))
    #         tortuosity_matrix = np.zeros((num_nodes, num_nodes))
    #         radius_matrix = np.zeros((num_nodes, num_nodes))
            
    #         # Populate the edge feature matrices based on edge_features data
    #         for edge_feature in edge_features[i]:
    #             node1 = edge_feature["node1"]
    #             node2 = edge_feature["node2"]
    #             length = edge_feature["length"]
    #             tortuosity = edge_feature["tortuosity"]
    #             radius = edge_feature["radius"]
                
    #             # Since the graph is undirected, populate both [node1, node2] and [node2, node1]
    #             length_matrix[node1, node2] = length
    #             length_matrix[node2, node1] = length
                
    #             tortuosity_matrix[node1, node2] = tortuosity
    #             tortuosity_matrix[node2, node1] = tortuosity
                
    #             radius_matrix[node1, node2] = radius
    #             radius_matrix[node2, node1] = radius

    #         # Convert adjacency and edge feature matrices to torch Tensors
    #         edge_index, edge_attr = dense_to_ind_val(
    #             adj[i], length_matrix=length_matrix, tortuosity_matrix=tortuosity_matrix, radius_matrix=radius_matrix
    #         )
            
    #         # Prepare 3D positions as node features
    #         node_pos = node_features[i]  # Assuming node_features[i] has shape (num_nodes, 3) for 3D positions
    #         node_pos = torch.tensor(node_pos, dtype=torch.float)
            
    #         data = Data(num_nodes=num_nodes, y=y[i], edge_index=edge_index, edge_attr=edge_attr, pos = node_pos)
            
    #         # added by Kaiyu, use pos as node feature
    #         data.x = node_pos
            
    #         data_list.append(data)

    #     if self.pre_filter is not None:
    #         data_list = [data for data in data_list if self.pre_filter(data)]

    #     if self.pre_transform is not None:
    #         data_list = [self.pre_transform(data) for data in data_list]
    #     else:
    #         data_list = [data for data in data_list]

    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices, num_nodes), self.processed_paths[0])

    def process(self):
        # if self.name in ['BRAVE', 'SKDementia', 'CROP', 'IPH', 'BRAINVASCULATURE']:
        if self.name:
            adj, edge_features, node_features, y = load_data(self.raw_dir)
            # check if y is a array with float64
            if y.dtype == np.float64:
                y = torch.tensor(y, dtype=torch.float)
            else:
                y = torch.LongTensor(y)
            adj = torch.Tensor(adj)
            num_graphs = adj.shape[0]
            num_nodes = adj.shape[1]

        data_list = []
        for i in range(num_graphs):
            # Prepare edge index and edge attributes directly
            edge_index = []
            edge_attr = []
            for edge_feature in edge_features[i]:
                node1 = edge_feature["node1"]
                node2 = edge_feature["node2"]

                # Add edge index
                edge_index.append([node1, node2])
                edge_index.append([node2, node1])  # Since the graph is undirected

                # Add edge attributes (length, tortuosity, radius)
                edge_attr.append([
                    edge_feature["length"],
                    edge_feature["tortuosity"],
                ])
                edge_attr.append([
                    edge_feature["length"],
                    edge_feature["tortuosity"],
                ])  # Same for the reverse edge

            # Convert edge data to tensors
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).T  # Shape [2, num_edges]
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)     # Shape [num_edges, num_features]

            # Prepare 3D positions as node features
            node_pos = node_features[i]  # Assuming node_features[i] has shape (num_nodes, 3) for 3D positions
            node_pos = torch.tensor(node_pos, dtype=torch.float)

            data = Data(num_nodes=num_nodes, y=y[i], edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, pos=node_pos)
            
            data.x = node_pos
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        else:
            data_list = [data for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, num_nodes), self.processed_paths[0])

    def _process(self):
        print('Processing...', file=sys.stderr)

        if files_exist(self.processed_paths):  # pragma: no cover
            print('Done!', file=sys.stderr)
            return

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        print('Done!', file=sys.stderr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name}()'
