import torch
from node_types import ArtistNode, TrackNode, GenreNode

class TrackSet:
    def __init__(self, tracks: list[TrackNode]):
        self.tracks = tracks

    def get_features(self) -> torch.tensor:
        stack_features = torch.stack([track.get_features() for track in self.tracks])
        return stack_features
    
    def get_features_gnn(self) -> torch.tensor:
        stack_features = torch.stack([artist.get_features_gnn() for artist in self.artists])
        return stack_features
    
class ArtistSet:
    def __init__(self, artists: list[ArtistNode]):
        self.artists = artists

    def get_features(self) -> torch.tensor:
        stack_features = torch.stack([artist.get_features() for artist in self.artists])
        return stack_features
    
    def get_features_gnn(self) -> torch.tensor:
        stack_features = torch.stack([artist.get_features_gnn() for artist in self.artists])
        return stack_features
    
class GenreSet:
    def __init__(self, genres: list[GenreNode]):
        self.genres = genres

    def get_features(self) -> torch.tensor:
        stack_features = torch.stack([genre.get_features() for genre in self.genres])
        return stack_features
    
    def get_features_gnn(self) -> torch.tensor:
        stack_features = torch.stack([artist.get_features_gnn() for artist in self.artists])
        return stack_features