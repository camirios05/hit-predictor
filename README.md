# Hit Predictor

Proyecto de **Predicción de Éxito Musical** desarrollado para la materia **Desarrollo de Aplicaciones para el Análisis de Datos**.

La aplicación utiliza técnicas de aprendizaje automático para analizar características acústicas de una canción y estimar si tiene potencial para convertirse en un éxito musical.

---

## Requisitos previos

Antes de comenzar, asegúrate de tener instalado:

- Python 3.8 o superior.
- pip (administrador de paquetes de Python).

Puedes descargar Python desde: https://www.python.org

---

## Instalación y ejecución local

### 1. Obtener el proyecto

Clona el repositorio o descarga los archivos del proyecto en una carpeta de tu equipo.

### 2. Abrir una terminal

Accede a la carpeta raíz del proyecto:

```bash
cd PROYECTO
```

### 3. Crear un entorno virtual (recomendado)

#### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar dependencias

Instala todas las librerías necesarias:

```bash
pip install -r requirements.txt
```

### 5. Colocar el dataset

Asegúrate de que el archivo:

```text
dataset.csv
```

se encuentre en la raíz del proyecto.

Al iniciar la aplicación, el modelo se entrenará automáticamente utilizando este conjunto de datos.

> Si el archivo no está disponible, la aplicación generará datos simulados para permitir pruebas de la interfaz web.

### 6. Iniciar el servidor

Ejecuta:

```bash
python app.py
```

Durante el arranque se mostrará en la terminal el proceso de entrenamiento del modelo.

Al finalizar aparecerá un mensaje similar a:

```text
* Running on http://127.0.0.1:5000
* Press CTRL+C to quit
```

### 7. Acceder a la aplicación

Abre tu navegador e ingresa a:

```text
http://127.0.0.1:5000
```

A partir de este momento podrás:

- Introducir el nombre de una canción.
- Seleccionar un género musical.
- Capturar sus características acústicas.
- Obtener una predicción instantánea generada por el modelo de inteligencia artificial.

---

## Detener la aplicación

Para apagar el servidor local, regresa a la terminal donde se está ejecutando y presiona:

```text
CTRL + C
```

---
## Glosario de generos musicales en el dataset
| acoustic | afrobeat | alt-rock | alternative | ambient | anime |
| black-metal | bluegrass | blues | brazilian | breakbeat | british |
| cantopop | chicago-house | children | chill | classical | club |
| comedy | country | dance | dancehall | death-metal | deep-house |
| disco | disney | drum-and-bass | dub | dubstep | edm |
| electro | electronic | emo | folk | forro | french |
| funk | garage | german | gospel | gothic | grindcore |
| groove | grunge | guitar | happy | hard-rock | hardcore |
| hardstyle | heavy-metal | hip-hop | honky-tonk | house | idm |
| indian | indie | indie-pop | industrial | iranian | j-dance |
| j-idol | j-pop | jazz | k-pop | kids | latin |
| latino | malay | mandopop | metal | metalcore | minimal-techno |
| movie | mpb | new-age | opera | pagode | party |
| piano | pop | punk | punk-rock | r-n-b | reggae |
| reggaeton | rock | rock-n-roll | romance | salsa | samba |
| sertanejo | show-tunes | singer-songwriter | ska | sleep | songwriter |
| soul | spanish | study | swedish | synth-pop | tango |
| techno | trance | trip-hop | turkish | world-music | |

## Glosario de métricas acústicas

### Tempo (BPM)

Velocidad de la canción medida en pulsaciones por minuto.

**Ejemplo:** `120.0 BPM`

---

### Danceability

Valor entre `0.0` y `1.0` que indica qué tan adecuada es una canción para bailar considerando ritmo, estabilidad y tempo.

---

### Energy

Valor entre `0.0` y `1.0` que representa la intensidad, actividad y dinamismo de la pista.

---

### Valence (Positividad)

Valor entre `0.0` y `1.0` que describe la emoción transmitida por la canción.

| Valor | Interpretación |
|---------|---------------|
| Cercano a 1.0 | Alegre, optimista, eufórica |
| Cercano a 0.0 | Triste, melancólica o tensa |

---

### Loudness

Volumen general de la pista medido en decibelios (dB).

Rango habitual:

```text
-60.0 dB a 0.0 dB
```

---

### Acousticness

Valor entre `0.0` y `1.0` que estima la probabilidad de que la canción sea predominantemente acústica.

| Valor | Interpretación |
|---------|---------------|
| Cercano a 1.0 | Principalmente acústica |
| Cercano a 0.0 | Mayor presencia de elementos electrónicos |



## Objetivo del proyecto

Desarrollar una aplicación web capaz de analizar métricas acústicas de una canción y predecir, mediante modelos de aprendizaje automático, si posee características similares a las de canciones exitosas dentro de la industria musical.