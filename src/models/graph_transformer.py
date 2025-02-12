import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, GINEConv, global_mean_pool, global_max_pool, global_add_pool

class GraphTransformer(nn.Module):
    def __init__(self, num_features, args, num_nodes, num_edge_features=None, num_classes=1):
        super(GraphTransformer, self).__init__()
        # Input encoders
        self.use_edge_attr = num_edge_features is not None and num_edge_features > 0
        # If using positional encodings or degree, increase input feature dim
        input_dim = num_features
        if hasattr(args, 'pe_dim') and args.pe_dim > 0:
            input_dim += args.pe_dim  # expecting positional enc of length pe_dim
            
        dropout = args.dropout if hasattr(args, 'dropout') else 0.0
        hidden_dim = args.hidden_dim
        # Node encoder with multiple layers but maintaining dimensions
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        if self.use_edge_attr:
            self.edge_encoder = nn.Sequential(
                nn.Linear(num_edge_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.edge_encoder = None
            
        # Define the local GNN conv to use inside GPSConv
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Pass hidden_dim as edge_dim since that's what our edge_encoder outputs
        local_conv = GINEConv(nn=mlp, edge_dim=hidden_dim) if self.use_edge_attr else GINEConv(nn=mlp)
        
        # Stack GPSConv layers
        self.layers = nn.ModuleList()
        num_layers = args.n_GNN_layers if hasattr(args, 'n_GNN_layers') else 2
        num_heads = args.num_heads if hasattr(args, 'num_heads') else 2
        
        for _ in range(num_layers):
            layer = GPSConv(channels=hidden_dim, conv=local_conv, heads=num_heads, 
                          dropout=dropout, attn_type='multihead')
            self.layers.append(layer)
            
        # Choose pooling function based on args
        pooling = args.pooling if hasattr(args, 'pooling') else "mean"
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            self.pool = global_add_pool  # "sum" pooling
            
        # Output MLP layers similar to GAT
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
        # Set regression flag
        self.regression = hasattr(args, 'regression') and args.regression
    
    def forward(self, x, edge_index, edge_attr, batch):
        
        # Encode inputs
        x = self.node_encoder(x)
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            
        # Process through GPS layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            
        # Global pooling
        x = self.pool(x, batch)
        
        # Output projection
        x = self.output_proj(x)
        
        return x
