import torch
from torch_geometric.data import HeteroData

class MusicGraph:
    def __init__(self, tracks_tensor: torch.tensor, artists_tensor: torch.tensor, genres_tensor: torch.tensor):
        self.tracks_data = tracks_tensor
        self.artists_data = artists_tensor
        self.genres_data = genres_tensor
        
        self._inititate_nodes()
        self._initiate_schema()

    def _inititate_nodes(self):
        self.data['track'].x = self.tracks_data.float()
        self.data['artist'].x = self.artists_data.float()
        self.data['genre'].x = self.genres_data.float()

    def _initiate_schema(self):
        self.node_types = ['track', 'artist', 'genre']
        self.edge_types = [
            ('artist', 'performed', 'track'),
            ('track', 'has_genre', 'genre'),
        ]
        self.metadata = (self.node_types, self.edge_types)
    
    def add_edges_performed(self, artist_idx: torch.LongTensor, track_idx: torch.LongTensor):
        element = torch.stack((artist_idx.long(), track_idx.long()), dim=0)
        self.data['artist', 'performed', 'track'].edge_index = element

    def add_edges_has_genre(self, track_idx: torch.LongTensor, genre_idx: torch.LongTensor):
        ei = torch.stack((track_idx.long(), genre_idx.long()), dim=0) 
        self.data['track','has_genre','genre'].edge_index = ei

    def get_graph(self) -> HeteroData:
        return self.data 

    def to(self, device: torch.device | str):
        self.data = self.data.to(device)
    
    def standardize_features_(self):
        for ntype in self.data.node_types:
            if 'x' in self.data[ntype]:
                X = self.data[ntype].x
                mean = X.mean(dim=0, keepdim=True)
                std  = X.std(dim=0, keepdim=True).clamp_min(1e-6)
                self.data[ntype].x = (X - mean) / std

    def validate(self) -> bool:
        d = self.data
        def ncount(node_type):
            return int(d[node_type].x.size(0)) if 'x' in d[node_type] else int(d[node_type].num_nodes)

        for node_type in ['track','artist','genre']:
            assert (('x' in d[node_type]) or ('num_nodes' in d[node_type])), f"Missing {node_type} nodes"
            if 'x' in d[node_type]:
                assert d[node_type].x.dtype.is_floating_point, f"{node_type}.x must be float"

        N_a, N_t, N_g = ncount('artist'), ncount('track'), ncount('genre')

        if ('artist','performed','track') in d.edge_types:
            ei = d['artist','performed','track'].edge_index
            assert ei.dtype == torch.long and ei.size(0) == 2
            assert int(ei[0].min()) >= 0 and int(ei[0].max()) < N_a
            assert int(ei[1].min()) >= 0 and int(ei[1].max()) < N_t

        if ('track','has_genre','genre') in d.edge_types:
            ei = d['track','has_genre','genre'].edge_index
            assert ei.dtype == torch.long and ei.size(0) == 2
            assert int(ei[0].min()) >= 0 and int(ei[0].max()) < N_t
            assert int(ei[1].min()) >= 0 and int(ei[1].max()) < N_g

        return True