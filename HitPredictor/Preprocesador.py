import pandas as pd

def limpiar_datos(df):
    if 'track_id' in df.columns:
        df = df.drop_duplicates(subset=['track_id'])
    
    num_cols = ['danceability', 'energy', 'tempo', 'loudness', 'acousticness']
    columnas = [col for col in num_cols if col in df.columns]
    if columnas:
        df[columnas] = df[columnas].fillna(df[columnas].median())
        
    if 'track_name' in df.columns and 'artists' in df.columns:
        df = df.dropna(subset=['track_name', 'artists'])
        df['track_name'] = df['track_name'].str.lower().str.strip()
        df['artists'] = df['artists'].str.lower().str.strip()
        
    if 'track_genre' in df.columns:
        df['track_genre'] = df['track_genre'].str.lower().str.strip()
        df['track_genre'] = df['track_genre'].fillna('desconocido')
        
    if 'duration_ms' in df.columns:
        df['duracion_minutos'] = df['duration_ms'] / 60000
        df = df.drop(columns=['duration_ms'])
        
    return df