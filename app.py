from flask import Flask, request, jsonify, render_template
import warnings


from HitPredictor.Entidades import MetricasAcusticas, GeneroMusical, Cancion
from HitPredictor.Predictor import CerebroPredictivo

warnings.filterwarnings('ignore')

app = Flask(__name__)


ia = CerebroPredictivo()


ia.entrenar_desde_csv(r"C:\Users\lenovo\Documents\ASemestre 4\Analisis\PROYECTO\dataset.csv")

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