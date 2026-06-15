from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

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

class GestorDatos:
    @staticmethod
    def limpiar_datos(df):
        if 'track_id' in df.columns:
            df = df.drop_duplicates(subset=['track_id'])
        num_cols = ['danceability', 'energy', 'tempo', 'loudness', 'acousticness']
        columnas_existentes = [col for col in num_cols if col in df.columns]
        if columnas_existentes:
            df[columnas_existentes] = df[columnas_existentes].fillna(df[columnas_existentes].median())
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

class CerebroPredictivo:
    def __init__(self):
        self.modelo_ml = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)
        columnas_numericas = ['tempo', 'danceability', 'energy', 'valence', 'loudness', 'acousticness']
        columna_categorica = ['track_genre']
        self.preprocesador = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), columnas_numericas),
                ('cat', OneHotEncoder(handle_unknown='ignore'), columna_categorica)
            ]
        )

    def entrenar_modelo(self, datos_entrenamiento):
        columnas = ['tempo', 'danceability', 'energy', 'valence', 'loudness', 'acousticness', 'track_genre', 'popularity']
        df_modelo = datos_entrenamiento.dropna(subset=columnas).copy()
        umbral_hit = 60
        df_modelo['es_hit'] = (df_modelo['popularity'] >= umbral_hit).astype(int)
        X_crudo = df_modelo[['tempo', 'danceability', 'energy', 'valence', 'loudness', 'acousticness', 'track_genre']]
        y = df_modelo['es_hit']
        X_procesado = self.preprocesador.fit_transform(X_crudo)
        X_train, X_test, y_train, y_test = train_test_split(X_procesado, y, test_size=0.2, random_state=42)
        self.modelo_ml.fit(X_train, y_train)
        return True

    def predecir_exito(self, demo):
        datos_dict = demo.obtener_diccionario_datos()
        datos_para_predecir_crudos = pd.DataFrame([datos_dict])
        datos_procesados = self.preprocesador.transform(datos_para_predecir_crudos)
        probabilidades = self.modelo_ml.predict_proba(datos_procesados)
        return probabilidades[0][1]
    
    def es_genero_conocido(self, genero_a_buscar):
        codificador = self.preprocesador.named_transformers_['cat']
        generos_oficiales = codificador.categories_[0]
        return genero_a_buscar in generos_oficiales

mi_ia = CerebroPredictivo()

def inicializar_ia():
    try:
        df_musica = pd.read_csv(r"C:\Users\lenovo\Documents\ASemestre 4\Analisis\PROYECTO\dataset.csv")
        df_musica = GestorDatos.limpiar_datos(df_musica)
    except FileNotFoundError:
        print("CSV no encontrado. Generando datos de prueba para la web...")
        np.random.seed(42)
        generos_posibles = ['pop', 'rock', 'hip-hop', 'electronic', 'reggaeton', 'jazz', 'classical']
        df_musica = pd.DataFrame({
            'tempo': np.random.randint(70, 180, 1000),
            'danceability': np.random.uniform(0.1, 1.0, 1000),
            'energy': np.random.uniform(0.1, 1.0, 1000),
            'valence': np.random.uniform(0.1, 1.0, 1000),
            'loudness': np.random.uniform(-15.0, 0.0, 1000),
            'acousticness': np.random.uniform(0.0, 1.0, 1000),
            'track_genre': [random.choice(generos_posibles) for _ in range(1000)], 
            'popularity': np.random.randint(0, 100, 1000)
        })
    mi_ia.entrenar_modelo(df_musica)

inicializar_ia()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.json
    
    mi_genero = GeneroMusical(datos['genero'])
    mis_metricas = MetricasAcusticas(
        tempo=datos['tempo'],
        danceability=datos['danceability'],
        energy=datos['energy'],
        valence=datos['valence'],
        loudness=datos['loudness'],
        acousticness=datos['acousticness']
    )
    mi_demo_cancion = Cancion(nombre=datos['nombre'], metricas=mis_metricas, genero=mi_genero)
    
    probabilidad = mi_ia.predecir_exito(mi_demo_cancion)
    conocido = mi_ia.es_genero_conocido(mi_demo_cancion.genero.nombre)
    
    if probabilidad > 0.70:
        veredicto = "Tiene un ADN altamente compatible con los éxitos."
    elif probabilidad > 0.45:
        veredicto = "Buen potencial, pero el terreno es competitivo para este género."
    else:
        veredicto = "Riesgo alto de pasar desapercibida. Sugiere reestructuración."

    return jsonify({
        "probabilidad": round(probabilidad * 100, 2),
        "genero": mi_demo_cancion.genero.nombre if conocido else "Omitido (No en dataset)",
        "veredicto": veredicto
    })

if __name__ == '__main__':
    app.run(debug=True)