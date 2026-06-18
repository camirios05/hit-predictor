class MetricasAcusticas:
    def __init__(self, tempo, danceability, energy, valence, loudness, acousticness):
        self.tempo = tempo
        self.danceability = danceability
        self.energy = energy
        self.valence = valence
        self.loudness = loudness
        self.acousticness = acousticness

class GeneroMusical:
    def __init__(self, nombre):
        self.nombre = nombre.lower().strip()

class Cancion:
    def __init__(self, nombre, metricas, genero):
        self.nombre = nombre
        self.metricas = metricas
        self.genero = genero 
        
    def obtener_diccionario_datos(self):
        return {
            'tempo': self.metricas.tempo,
            'danceability': self.metricas.danceability,
            'energy': self.metricas.energy,
            'valence': self.metricas.valence,
            'loudness': self.metricas.loudness,
            'acousticness': self.metricas.acousticness,
            'track_genre': self.genero.nombre 
        }