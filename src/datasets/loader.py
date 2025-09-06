import numpy as np
import torch
from dataclasses import dataclass
import pandas as pd
from src.datasets.node_types import ArtistNode, TrackNode, GenreNode
from torch_geometric.data import HeteroData
from src.datasets.set_types import ArtistSet, TrackSet, GenreSet

@dataclass
class GraphDataInput:
    artists: list[dict]
    tracks: list[dict]
    genres: list[dict]

    artist_to_track: list[tuple[int, int]]
    track_to_genre: list[tuple[int, int]]


def _fill_missing_cols(data: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in data.columns:
            data[col] = torch.nan

def _safe_float(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        try:
            data[col] = data[col].astype(float)
        except (ValueError, TypeError):
            data[col] = torch.nan
    return data

def csv_to_dicts(
        artist_csv: str, 
        track_csv: str, 
        genre_csv: str, 
        features_csv: str
    ) -> GraphDataInput:

    artists = pd.read_csv(artist_csv).to_dict(orient='records')
    tracks = pd.read_csv(track_csv).to_dict(orient='records')
    genres = pd.read_csv(genre_csv).to_dict(orient='records')
    features = pd.read_csv(features_csv).to_dict(orient='records')

    tracks_full = tracks.merge(features, on='track_id', how='left')

    _fill_missing_cols(artists, ArtistNode.feature_list)
    _fill_missing_cols(genres, GenreNode.feature_list)
    _fill_missing_cols(tracks_full, TrackNode.feature_list) 

    artist_filled = _safe_float(artists, ArtistNode.feature_gnn)
    genre_filled = _safe_float(genres, GenreNode.feature_gnn)
    track_filled = _safe_float(tracks_full, TrackNode.feature_gnn)

    artists = artist_filled[ArtistNode.feature_list].to_dict(orient='records')
    tracks  = track_filled[TrackNode.feature_list].to_dict(orient='records')
    genres  = genre_filled[GenreNode.feature_list].to_dict(orient='records')

    artist_track_tuples = []
    for track in track_filled:
        if pd.notna(track.get('artist_id')) and pd.notna(track.get('track_id')):
            artist_track_tuples.append((int(track['artist_id']), int(track['track_id'])))

    track_genre_tuples = []
    for track in track_filled:
        if pd.notna(track.get('track_id')) and pd.notna(track.get('genres_ids')):
            genre_ids = track['genres_ids']
            for genre_id in genre_ids:
                if genre_id.isdigit():
                    track_genre_tuples.append((int(track['track_id']), int(genre_id)))


    return GraphDataInput(
        artists = artists,
        tracks = tracks,
        genres = genres,
        artist_to_track = artist_track_tuples,
        track_to_genre = track_genre_tuples
    )

    
def build_set_structires(data: GraphDataInput) -> tuple[ArtistSet, TrackSet, GenreSet]:
    artist_nodes = [ArtistNode(features=artist) for artist in data.artists]
    track_nodes = [TrackNode(features=track) for track in data.tracks]
    genre_nodes = [GenreNode(features=genre) for genre in data.genres]

    artist_set = ArtistSet(artist_nodes)
    track_set = TrackSet(track_nodes)
    genre_set = GenreSet(genre_nodes)

    return artist_set, track_set, genre_set

def build_hetero_data(
        data: GraphDataInput, 
        artist_set: ArtistSet, 
        track_set: TrackSet, 
        genre_set: GenreSet
    ) -> HeteroData:
    
    hetero_data = HeteroData()

    hetero_data['artist'].x = artist_set.get_features_gnn()
    hetero_data['track'].x = track_set.get_features_gnn()
    hetero_data['genre'].x = genre_set.get_features_gnn()

    artist_id_to_index = {int(artist.features['artist_id']): idx for idx, artist in enumerate(artist_set.artists)}
    track_id_to_index = {int(track.features['track_id']): idx for idx, track in enumerate(track_set.tracks)}
    genre_id_to_index = {int(genre.features['genre_id']): idx for idx, genre in enumerate(genre_set.genres)}

    artist_to_track_edges = [
        (artist_id_to_index[src], track_id_to_index[dst]) 
        for src, dst in data.artist_to_track 
        if src in artist_id_to_index and dst in track_id_to_index
    ]
    if artist_to_track_edges:
        hetero_data['artist', 'creates', 'track'].edge_index = torch.tensor(artist_to_track_edges).t().contiguous()
    else:
        hetero_data['artist', 'creates', 'track'].edge_index = torch.empty((2,0), dtype=torch.long)

    track_to_genre_edges = [
        (track_id_to_index[src], genre_id_to_index[dst]) 
        for src, dst in data.track_to_genre 
        if src in track_id_to_index and dst in genre_id_to_index
    ]
    if track_to_genre_edges:
        hetero_data['track', 'belongs_to', 'genre'].edge_index = torch.tensor(track_to_genre_edges).t().contiguous()
    else:
        hetero_data['track', 'belongs_to', 'genre'].edge_index = torch.empty((2,0), dtype=torch.long)

    return hetero_data

    