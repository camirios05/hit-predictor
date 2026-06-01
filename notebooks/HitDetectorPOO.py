import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import warnings
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')


# ARQUITECTURA POO (MÉTRICAS Y CANCIÓN)

class MetricasAcusticas:
    def __init__(self, tempo, danceability, energy, valence, loudness, acousticness):
        self.tempo = tempo
        self.danceability = danceability
        self.energy = energy
        self.valence = valence
        self.loudness = loudness
        self.acousticness = acousticness

    def obtener_vector_caracteristicas(self):
        """Retorna las métricas en el orden exacto que espera el modelo."""
        return [self.tempo, self.danceability, self.energy, self.valence, self.loudness, self.acousticness]

class Cancion:
    def __init__(self, nombre, metricas):
        self.nombre = nombre
        self.metricas = metricas


# Limpieza de datos
class GestorDatos:
    @staticmethod
    def limpiar_datos(df):
        # 1. Eliminar duplicados
        df = df.drop_duplicates(subset=['id_cancion'])
        
        # 2. Manejo de valores nulos
        # Llenar nulos en variables numéricas con la mediana
        num_cols = ['danceability', 'energy', 'tempo_bpm', 'loudness', 'acousticness']
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        # Eliminar filas si falta el nombre o el artista
        df = df.dropna(subset=['nombre', 'artista'])
        
        # 3. Normalización de texto (para cruzar datos de Spotify y Billboard)
        df['nombre'] = df['nombre'].str.lower().str.strip()
        df['artista'] = df['artista'].str.lower().str.strip()
        
        # 4. Transformación de variables
        # Convertir duración de milisegundos a minutos
        if 'duracion_ms' in df.columns:
            df['duracion_minutos'] = df['duracion_ms'] / 60000
            df = df.drop(columns=['duracion_ms'])
        
        # 5. Escalar variables numéricas para el modelo predictivo
        scaler = MinMaxScaler()
        variables_a_escalar = ['danceability', 'energy', 'tempo_bpm', 'valence', 'duracion_minutos']
        df[variables_a_escalar] = scaler.fit_transform(df[variables_a_escalar])
        
        return df


# EL CEREBRO PREDICTIVO

class CerebroPredictivo:
    def __init__(self):
        self.modelo_ml = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15,
            class_weight='balanced', 
            random_state=42
        )
        self.escalador = MinMaxScaler()
        
    def entrenar_modelo(self, datos_entrenamiento):
        print("\nENTRENANDO CEREBRO PREDICTIVO")
        
        columnas = ['tempo', 'danceability', 'energy', 'valence', 'loudness', 'acousticness', 'popularity']
        
        if not all(col in datos_entrenamiento.columns for col in columnas):
            print("Error: Faltan columnas en el dataset.")
            return False

        df_modelo = datos_entrenamiento.dropna(subset=columnas).copy()
        
        umbral_hit = 60
        df_modelo['es_hit'] = (df_modelo['popularity'] >= umbral_hit).astype(int)
        
        X_crudo = df_modelo[['tempo', 'danceability', 'energy', 'valence', 'loudness', 'acousticness']]
        y = df_modelo['es_hit']
        
        X_escalado = self.escalador.fit_transform(X_crudo)
        X = pd.DataFrame(X_escalado, columns=X_crudo.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.modelo_ml.fit(X_train, y_train)
        
        precision = accuracy_score(y_test, self.modelo_ml.predict(X_test))
        print(f"Modelo entrenado. Precisión base: {precision * 100:.2f}%")
        return True

    def predecir_exito(self, demo):
        vector_datos = demo.metricas.obtener_vector_caracteristicas()
        nombres_columnas = ['tempo', 'danceability', 'energy', 'valence', 'loudness', 'acousticness']
        datos_para_predecir_crudos = pd.DataFrame([vector_datos], columns=nombres_columnas)
        datos_escalados = self.escalador.transform(datos_para_predecir_crudos)
        probabilidades = self.modelo_ml.predict_proba(datos_escalados)
        return probabilidades[0][1]


#  WEB SCRAPER DE BILLBOARD

def scrape_billboard_top_10():
    url = "https://www.billboard.com/charts/hot-100/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    canciones = []
    items = soup.find_all('div', class_='o-chart-results-list-row-container')
    
    for item in items[:10]:
        titulo = item.find('h3', id='title-of-a-story').text.strip()
        artista = item.find('span', class_='c-label').text.strip()
        canciones.append({'titulo': titulo, 'artista': artista})
        
    return pd.DataFrame(canciones)


#  ZONA DE PRUEBA INTERACTIVA (MODO PRODUCTOR)

if __name__ == "__main__":
    
    ruta_archivo = 'dataset.csv' 
    try:
        # Carga del archivo original
        df_musica = pd.read_csv(r"C:\Users\lenovo\Documents\ASemestre 4\Analisis\PROYECTO\dataset.csv")
        
        # Llamada al Gestor de Datos para limpiar y transformar el dataset
        print("\n[INFO] Iniciando proceso de limpieza de datos...")
        df_musica = GestorDatos.limpiar_datos(df_musica)
        print("[INFO] Dataset limpiado y escalado correctamente.")
        
    except FileNotFoundError:
        print(" No se encontró el archivo CSV. Generando datos de prueba...")
        # Generador de respaldo rápido por si el CSV falla
        import numpy as np
        np.random.seed(42)
        df_musica = pd.DataFrame({
            'tempo': np.random.randint(70, 180, 1000),
            'danceability': np.random.uniform(0.1, 1.0, 1000),
            'energy': np.random.uniform(0.1, 1.0, 1000),
            'valence': np.random.uniform(0.1, 1.0, 1000),
            'loudness': np.random.uniform(-15.0, 0.0, 1000),
            'acousticness': np.random.uniform(0.0, 1.0, 1000),
            'popularity': np.random.randint(0, 100, 1000)
        })

    mi_ia = CerebroPredictivo()
    
    # Llamamos al método actualizado
    entrenamiento_exitoso = mi_ia.entrenar_modelo(df_musica)
    
    if entrenamiento_exitoso:
        print("\nINICIANDO MODO PRODUCTOR")
        print("Bienvenido al laboratorio. Ingresa las métricas acústicas de tu demo.")
        
        try:
            nombre_demo = input(" Nombre de tu pista/demo: ")
            
            print("\nIntroduce los valores numéricos:")
            demo_tempo = float(input(" - Tempo (BPM, ej. 120.0): "))
            demo_danceability = float(input(" - Danceability (0.0 a 1.0): "))
            demo_energy = float(input(" - Energía (0.0 a 1.0): "))
            demo_valence = float(input(" - Positividad / Valence (0.0 a 1.0): "))
            demo_loudness = float(input(" - Loudness/Volumen (en dB, ej. -4.5): ")) 
            demo_acousticness = float(input(" - Acousticness (0.0 a 1.0, ej. 0.01): "))
            
            print(f"\nProcesando el ADN Acústico de '{nombre_demo}'...")
            
            # Encapsulamos los datos en objetos (POO)
            mis_metricas = MetricasAcusticas(
                tempo=demo_tempo, 
                danceability=demo_danceability, 
                energy=demo_energy, 
                valence=demo_valence, 
                loudness=demo_loudness, 
                acousticness=demo_acousticness
            )
            mi_demo_cancion = Cancion(nombre=nombre_demo, metricas=mis_metricas)
            
            # Pasamos el objeto Cancion completo al Cerebro
            probabilidad = mi_ia.predecir_exito(mi_demo_cancion)
            
            print("\nRESULTADO DE HIT PREDICTOR ")
            print(f"Probabilidad matemática de ser un Éxito Viral: {probabilidad * 100:.2f}%")
            
            if probabilidad > 0.70:
                print(" Veredicto: Tiene un ADN altamente compatible con los éxitos.")
            elif probabilidad > 0.45:
                print(" Veredicto: Buen potencial, pero el terreno es competitivo.")
            else:
                print(" Veredicto: Riesgo alto de pasar desapercibida. Sugiere reestructuración.")

        except ValueError:
            print("\n Error: Por favor ingresa solo números válidos. Usa punto (.) para decimales.")
            
    print("\nCONTRASTE EN TIEMPO REAL: BILLBOARD HOT 100 ")
    try:
        df_top10 = scrape_billboard_top_10()
        print(df_top10.head(5).to_string(index=False))
    except Exception as e:
        print("Error leyendo Billboard:", e)