import torch

class MusicNode():
    feature_list: list[str] = []

    def __init__(self, features: dict):
        self.features = features

    def get_features(self) -> dict:
        vals = torch.tensor(self.features.get(key, torch.nan) for key in self.feature_list)
        return vals
    
    
class ArtistNode(MusicNode):
    feature_list = ['artist_id', 'artist_name', 'artist_year', 'artist_location','artist_bio']
    feature_gnn = ['artist_year']

    def __init__(self, features: dict):
        super().__init__(features)
    
    def get_features_gnn(self) -> torch.tensor:
        vals = torch.tensor([self.features.get(key, torch.nan) for key in self.feature_gnn], dtype = torch.float32)
        return vals
    

class TrackNode(MusicNode):
    feature_list = [
        'track_id', 'album_date_release', 'album_listens', 'album_title',
        'tracks_in_album', 'artist_id', 'track_date_release', 'duration',
        'genres_ids', 'language', 'listens', 'track_title',
        'acousticness','danceability','energy','instrumentalness','liveness','speechiness','tempo','valence',
    ]
    feature_gnn = [
        'album_listens', 'tracks_in_album', 'duration', 'listens','acousticness',
        'danceability','energy','instrumentalness','liveness','speechiness','tempo','valence',
    ]

    def __init__(self, features: dict):
        super().__init__(features)
    
    def get_features_gnn(self) -> torch.tensor:
        vals = torch.tensor([self.features.get(key, torch.nan) for key in self.feature_gnn], dtype = torch.float32)
        return vals
    
    
class GenreNode(MusicNode):
    feature_list = [
        'genre_id', '#tracks', 'parent', 'title', 'top_level'
    ]
    feature_gnn = ['#tracks']

    def __init__(self, features: dict):
        super().__init__(features)
    
    def get_features_gnn(self) -> torch.tensor:
        vals = torch.tensor([self.features.get(key, torch.nan) for key in self.feature_gnn], dtype = torch.float32)
        return vals
    