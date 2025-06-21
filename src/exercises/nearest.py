import numpy as np
import json
import os

# Estadísticas reales (referencia)
estadisticas_reales = {
    "1": {"media": 77.9, "mediana": 87, "desviacion": 26.08},
    "2": {"media": 40.9, "mediana": 35, "desviacion": 29.49},
    "3": {"media": 71.7, "mediana": 70, "desviacion": 16.88},
    "4": {"media": 46.9, "mediana": 40, "desviacion": 22.48},
    "5": {"media": 90.7, "mediana": 100, "desviacion": 15.87},
    "6": {"media": 72.9, "mediana": 100, "desviacion": 34.05}
}

# Transforma JSON plano a formato anidado por lambda y k
def transformar_estadisticas(json_viejo):
    nuevo = {}
    for clave, valores in json_viejo.items():
        partes = clave.split("_")
        lambda_key = partes[0]
        k_key = partes[1]
        if lambda_key not in nuevo:
            nuevo[lambda_key] = {}
        nuevo[lambda_key][k_key] = valores
    return nuevo

# Distancia entre dos sets de estadísticas
def distancia(e1, e2):
    return ((e1['media'] - e2['media']) ** 2 +
            (e1['mediana'] - e2['mediana']) ** 2 )/2
            #+(e1['desviacion'] - e2['desviacion']) ** 2)/ 3

# Encuentra el mejor lambda y k para un juego
def encontrar_mejor_match(json_juego, stats_reales):
    mejor_dist = float('inf')
    mejor_lambda, mejor_k = None, None

    for lambda_key in json_juego:
        for k_key in json_juego[lambda_key]:
            stats = json_juego[lambda_key][k_key]
            d = distancia(stats, stats_reales)
            if d < mejor_dist:
                mejor_dist = d
                mejor_lambda, mejor_k = lambda_key, k_key

    return mejor_lambda, mejor_k, mejor_dist

# Procesamiento general
resultados = {}

for i in range(1, 7):  # Juegos del 1 al 6
    input_file = f"/home/pop/Documents/Carrera/tomacode/decisionmaking-code-py/src/exercises/graficos/estadisticas_game_h_{i}.json"

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Datos cargados exitosamente desde: {input_file}")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {input_file}")
        continue
    except json.JSONDecodeError:
        print(f"❌ Error: El archivo {input_file} no tiene formato JSON válido")
        continue

    data_anidado = transformar_estadisticas(data)
    lambda_key, k_key, dist = encontrar_mejor_match(data_anidado, estadisticas_reales[str(i)])

    resultados[str(i)] = {
        "lambda": lambda_key,
        "k": k_key,
        "distancia": dist
    }

# Mostrar resultados finales
print("\n🎯 Resultados finales:")
print(json.dumps(resultados, indent=2))
