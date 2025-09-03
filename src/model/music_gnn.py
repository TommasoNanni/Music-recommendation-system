import torch
import torch.nn as nn
from encoder import NodeEncoder
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import HeteroData
import torch.nn.functional as F

class MusicGNN(torch.nn.Module):
    def __init__(
        self,
        metadata: tuple,
        input_dims: dict,
        num_nodes: dict,
        hidden_dimension: int,
        output_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_encoder = NodeEncoder(input_dims, num_nodes, hidden_dimension)
        self.node_types, self.edge_types = metadata

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = torch.nn.ModuleDict()
            conv['artist__performed__track'] = GATv2Conv(
                (-1, -1),
                hidden_dimension,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
            )
            conv['track__has_genre__genre'] = GATv2Conv(
                (-1, -1),
                hidden_dimension,
                heads=num_heads,
                concat = False,
                dropout=dropout,
                add_self_loops=False,
            )
            self.convs.append(conv)
            self.linears = nn.ModuleDict({ntype: nn.Linear(hidden_dimension, output_dim) for ntype in self.node_types})
            self.activation = nn.ELU()

    def forward(self, data: HeteroData) -> dict:
        encoded = self.node_encoder(data.x_dict)
        for conv in self.convs:
            hidden = conv(encoded, data.edge_index_dict)
            hidden = {key: self.act(val) for key, val in hidden.items()}

        out = {node: F.normalize(self.linears[node](hid), p=2, dim=-1) for node, hid in hidden.items()}
        return out
