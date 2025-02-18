import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, MessagePassing
from torch.nn import Parameter
import numpy as np
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot, zeros
from typing import Tuple
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch import nn
import torch_geometric
import math
import pdb

class MPGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, edge_emb_dim: int, gcn_mp_type: str, bucket_sz: float, 
                 normalize: bool = True, bias: bool = True, num_edge_features: int=None):
        super(MPGCNConv, self).__init__(in_channels=in_channels, out_channels=out_channels, aggr='add')

        self.edge_emb_dim = edge_emb_dim
        self.gcn_mp_type = gcn_mp_type
        self.bucket_sz = bucket_sz
        self.bucket_num = math.ceil(2.0/self.bucket_sz)
        if gcn_mp_type == "bin_concate":
            self.edge2vec = nn.Embedding(self.bucket_num, edge_emb_dim)

        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        
        input_dim = out_channels
        if gcn_mp_type == "bin_concate" or gcn_mp_type == "edge_weight_concate":
            input_dim = out_channels + edge_emb_dim
        elif gcn_mp_type == "edge_node_concate":
            input_dim = out_channels*2 + num_edge_features if num_edge_features is not None else out_channels*2 + 1
        elif gcn_mp_type == "node_concate":
            input_dim = out_channels*2
        self.edge_lin = torch.nn.Linear(input_dim, out_channels)
        
        # Transform edge features to a single weight if needed
        self.edge_transform = torch.nn.Linear(num_edge_features, 1) if num_edge_features is not None else None

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def message(self, x_i, x_j, edge_weight):
        # Transform edge features to weights if transform exists
        if self.edge_transform is not None:
            edge_weight_transformed = self.edge_transform(edge_weight).unsqueeze(-1)
        else:
            edge_weight_transformed = edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight
        
        if self.gcn_mp_type == "weighted_sum": 
            # use edge_weight as multiplier
            msg = edge_weight_transformed * x_j
        elif self.gcn_mp_type == "bin_concate":
            # concat xj and learned bin embedding
            bucket = torch.div(edge_weight_transformed + 1, self.bucket_sz, rounding_mode='trunc').int()
            bucket = torch.clamp(bucket, 0, self.bucket_num-1)
            msg = torch.cat([x_j, self.edge2vec(bucket).squeeze()], dim=1)
            msg = self.edge_lin(msg) 
        elif self.gcn_mp_type == "edge_weight_concate":
            # concat xj and tiled edge attr
            msg = torch.cat([x_j, edge_weight_transformed.repeat(1, self.edge_emb_dim)], dim=1)
            msg = self.edge_lin(msg) 
        elif self.gcn_mp_type == "edge_node_concate": 
            # concat xi, xj and full edge features
            msg = torch.cat([x_i, x_j, edge_weight], dim=1)
            msg = self.edge_lin(msg)
        elif self.gcn_mp_type == "node_concate":
            # concat xi and xj
            msg = torch.cat([x_i, x_j], dim=1)
            msg = self.edge_lin(msg)
        else:
            raise ValueError(f'Invalid message passing variant {self.gcn_mp_type}')
        return msg

        
class GCN(torch.nn.Module):
    def __init__(self, input_dim, args, num_nodes, num_edge_features=None, num_classes=2):
        super(GCN, self).__init__()
        self.activation = torch.nn.ReLU()
        self.convs = torch.nn.ModuleList()
        self.pooling = args.pooling
        self.num_nodes = num_nodes

        hidden_dim = args.hidden_dim
        num_layers = args.n_GNN_layers
        edge_emb_dim = args.edge_emb_dim
        gcn_mp_type = args.gcn_mp_type
        bucket_sz = args.bucket_sz
        gcn_input_dim = input_dim

        for i in range(num_layers-1):
            conv = torch_geometric.nn.Sequential('x, edge_index, edge_attr', [
                (MPGCNConv(gcn_input_dim, hidden_dim, edge_emb_dim, gcn_mp_type, bucket_sz, 
                          normalize=True, bias=True, num_edge_features=num_edge_features),'x, edge_index, edge_attr -> x'),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            gcn_input_dim = hidden_dim
            self.convs.append(conv)

        input_dim = 0

        if self.pooling == "concat":
            node_dim = hidden_dim // 4  # Scale based on hidden_dim
            conv = torch_geometric.nn.Sequential('x, edge_index, edge_attr', [
                (MPGCNConv(hidden_dim, hidden_dim, edge_emb_dim, gcn_mp_type, bucket_sz, 
                          normalize=True, bias=True, num_edge_features=num_edge_features),'x, edge_index, edge_attr -> x'),
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(64, node_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(node_dim)
            ])
            input_dim = node_dim*num_nodes

        elif self.pooling == 'sum' or self.pooling == 'mean':
            node_dim = hidden_dim  # Use the provided hidden_dim
            input_dim = node_dim
            conv = torch_geometric.nn.Sequential('x, edge_index, edge_attr', [
                (MPGCNConv(hidden_dim, hidden_dim, edge_emb_dim, gcn_mp_type, bucket_sz, 
                          normalize=True, bias=True, num_edge_features=num_edge_features),'x, edge_index, edge_attr -> x'),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(node_dim)
            ])

        self.convs.append(conv)

        self.fcn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        z = x
        edge_attr = torch.abs(edge_attr)
   
        # ## Hack for the edge_attr, only require tortuosity
        # edge_attr = edge_attr.view(x.shape[0], x.shape[1], 4)[:, :, 2].reshape(-1)

        # # check x, edge_index, edge_attr whether float32
        z = z.float()
        edge_attr = edge_attr.float()
   
        for i, conv in enumerate(self.convs):
            # bz*nodes, hidden
            z = conv(z, edge_index, edge_attr)

        if self.pooling == "concat":
            z = z.reshape((z.shape[0]//self.num_nodes, -1))
        elif self.pooling == 'sum':
            z = global_add_pool(z,  batch)  # [N, F]
        elif self.pooling == 'mean':
            z = global_mean_pool(z, batch)  # [N, F]

        out = self.fcn(z)
        return out

        

