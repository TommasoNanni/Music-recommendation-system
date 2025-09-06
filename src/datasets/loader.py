import numpy as np
import torch
from dataclasses import dataclass
import pandas as pd
from src.datasets.node_types import ArtistNode, TrackNode, GenreNode

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
        artists = artist_filled.to_dict(orient='records'),

    