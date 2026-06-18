import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


from HitPredictor.Preprocessing import limpiar_datos

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

    def entrenar_desde_csv(self, ruta_csv):
        try:
            df_musica = pd.read_csv(ruta_csv)
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
        
        self.ml(df_musica)

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
        datos = demo.obtener_diccionario_datos()
        datos_crudos = pd.DataFrame([datos])
        datos_final = self.preprocesador.transform(datos_crudos)
        prob = self.modelo_ml.predict_proba(datos_final)
        return prob[0][1]
    
    def genero(self, gen):
        codificador = self.preprocesador.named_transformers_['cat']
        generos = codificador.categories_[0]
        return gen in generos