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
        
    def dicdatos(self):
        return {
            'tempo': self.metricas.tempo,
            'danceability': self.metricas.danceability,
            'energy': self.metricas.energy,
            'valence': self.metricas.valence,
            'loudness': self.metricas.loudness,
            'acousticness': self.metricas.acousticness,
            'track_genre': self.genero.nombre 
        }

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

class CerebroPredictivo:
    def __init__(self):
        self.modelo_ml = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)
        colnum = ['tempo', 'danceability', 'energy', 'valence', 'loudness', 'acousticness']
        colcat = ['track_genre']
        self.preprocesador = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), colnum),
                ('cat', OneHotEncoder(handle_unknown='ignore'), colcat)
            ]
        )

    def ml(self, datose):
        columnas = ['tempo', 'danceability', 'energy', 'valence', 'loudness', 'acousticness', 'track_genre', 'popularity']
        df_modelo = datose.dropna(subset=columnas).copy()
        eshit = 60
        df_modelo['es_hit'] = (df_modelo['popularity'] >= eshit).astype(int)
        xcrudo = df_modelo[['tempo', 'danceability', 'energy', 'valence', 'loudness', 'acousticness', 'track_genre']]
        y = df_modelo['es_hit']
        xfinal = self.preprocesador.fit_transform(xcrudo)
        X_train, X_test, y_train, y_test = train_test_split(xfinal, y, test_size=0.2, random_state=42)
        self.modelo_ml.fit(X_train, y_train)
        return True

    def eshit(self, demo):
        datos = demo.dicdatos()
        datos_crudos = pd.DataFrame([datos])
        datos_final = self.preprocesador.transform(datos_crudos)
        prob = self.modelo_ml.predict_proba(datos_final)
        return prob[0][1]
    
    def genero(self, gen):
        codificador = self.preprocesador.named_transformers_['cat']
        generos = codificador.categories_[0]
        return gen in generos

ia = CerebroPredictivo()

def iastart():
    try:
        df_musica = pd.read_csv("./dataset.csv")
        df_musica = limpiar_datos(df_musica)
    except FileNotFoundError:
        print("CSV no encontrado. Generando datos de prueba para la web...")
        np.random.seed(42)
        generosdata = ['pop', 'rock', 'hip-hop', 'electronic', 'reggaeton', 'jazz', 'classical']
        df_musica = pd.DataFrame({
            'tempo': np.random.randint(70, 180, 1000),
            'danceability': np.random.uniform(0.1, 1.0, 1000),
            'energy': np.random.uniform(0.1, 1.0, 1000),
            'valence': np.random.uniform(0.1, 1.0, 1000),
            'loudness': np.random.uniform(-15.0, 0.0, 1000),
            'acousticness': np.random.uniform(0.0, 1.0, 1000),
            'track_genre': [random.choice(generosdata) for _ in range(1000)], 
            'popularity': np.random.randint(0, 100, 1000)
        })
    ia.ml(df_musica)

iastart()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.json
    
    migenero = GeneroMusical(datos['genero'])
    mismetricas = MetricasAcusticas(
        tempo=datos['tempo'],
        danceability=datos['danceability'],
        energy=datos['energy'],
        valence=datos['valence'],
        loudness=datos['loudness'],
        acousticness=datos['acousticness']
    )
    demo = Cancion(nombre=datos['nombre'], metricas=mismetricas, genero=migenero)
    
    probabilidad = ia.eshit(demo)
    conocido = ia.genero(demo.genero.nombre)
    
    if probabilidad > 0.70:
        decision = "Alta probabilidad de ser hit"
    elif probabilidad > 0.45:
        decision = "Potencial hit"
    else:
        decision = "Baja probabilidad de ser hit"

    return jsonify({
        "probabilidad": round(probabilidad * 100, 2),
        "genero": demo.genero.nombre if conocido else "Omitido (No en dataset)",
        "decision": decision
    })

if __name__ == '__main__':
    app.run(debug=True)