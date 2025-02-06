import torch
from collections import defaultdict
import numpy as np 
from itertools import permutations
from torch_geometric.utils import to_dense_adj
from torch.nn import functional as F


class BrainNN(torch.nn.Module):
    def __init__(self, args, gnn, discriminator=lambda x, y: x @ y.t()):
        super(BrainNN, self).__init__()
        self.gnn = gnn
        self.pooling = args.pooling
        self.discriminator = discriminator
        self.regression = getattr(args, 'regression', False)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        g = self.gnn(x, edge_index, edge_attr, batch)
        
        if not self.regression:
            # For classification, apply log_softmax
            g = F.log_softmax(g, dim=-1)
        # For regression (FRS prediction), return raw logits for BCE with logits
        return g
