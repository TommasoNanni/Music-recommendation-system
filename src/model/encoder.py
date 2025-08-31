import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.data import HeteroData

class NodeEncoder(nn.Module):
    def __init__(self, input_dims: int, num_nodes: int, hidden_dimensions: int):
        super().__init__()
        self.enc = nn.ModuleDict()
        for node_type, input_dim in input_dims.items():
            if input_dim and input_dim > 0:
                self.enc[node_type] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dimensions),
                    nn.ReLU(inplace=True)
                )
            else:
                self.enc[node_type] = nn.Embedding(num_nodes[node_type], hidden_dimensions)

    def forward(self, x_dict: dict) -> dict:
        hid = {}
        for node_type, enc in self.enc.items():
            x = x_dict.get(node_type)
            hid[node_type] = enc(x)
        return hid