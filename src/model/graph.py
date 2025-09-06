import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from dataclasses import dataclass


class MusicGraph:
    def __init__(self, tracks_tensor: torch.tensor, artists_tensor: torch.tensor, genres_tensor: torch.tensor):
        self.tracks_data = tracks_tensor
        self.artists_data = artists_tensor
        self.genres_data = genres_tensor
        self.data = HeteroData()
        self._inititate_nodes()
        self._initiate_schema()

    def _inititate_nodes(self):
        self.data['track'].x = self.tracks_data
        self.data['artist'].x = self.artists_data
        self.data['genre'].x = self.genres_data

    def _initiate_schema(self):
        self.node_types = ['track', 'artist', 'genre']
        self.edge_types = [
            ('artist', 'performed', 'track'),
            ('track', 'has_genre', 'genre'),
        ]
        self.metadata = (self.node_types, self.edge_types)

    def get_graph(self) -> HeteroData:
        return self.graph
    
    def add_edges_performed(self, artist_idx: torch.LongTensor, track_idx: torch.LongTensor):
        element = torch.stack((artist_idx.long(), track_idx.long()), dim=0)
        self.data['artist', 'performed', 'track'].edge_index = element

    def add_edges_has_genre(self, track_idx: torch.LongTensor, genre_idx: torch.LongTensor):
        ei = torch.stack((track_idx.long(), genre_idx.long()), dim=0) 
        self.data['track','has_genre','genre'].edge_index = ei
