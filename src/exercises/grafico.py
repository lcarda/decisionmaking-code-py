import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Cargar datos desde archivo JSON
input_file = "/home/pop/Documents/Carrera/tomacode/decisionmaking-code-py/src/exercises/datos/hierarchical_softmax_game_5.json"

try:
    with open(input_file, 'r') as f:
        data = json.load(f)
    print(f"✅ Datos cargados exitosamente desde: {input_file}")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo {input_file}")
    exit()
except json.JSONDecodeError:
    print(f"❌ Error: El archivo {input_file} no tiene formato JSON válido")
    exit()

def calcular_estadisticas(estrategias, probabilidades):
    """Calcula media, mediana y desviación estándar de una distribución discreta"""
    # Convertir a arrays numpy
    estrategias = np.array(estrategias)
    probabilidades = np.array(probabilidades)
    
    # Media (valor esperado)
    media = np.sum(estrategias * probabilidades)
    
    # Mediana (aproximada para distribución discreta)
    cumsum = np.cumsum(probabilidades)
    mediana_idx = np.where(cumsum >= 0.5)[0][0]
    mediana = estrategias[mediana_idx]
    
    # Desviación estándar
    varianza = np.sum(((estrategias - media) ** 2) * probabilidades)
    desviacion = np.sqrt(varianza)
    
    return media, mediana, desviacion

def crear_distribucion_completa(estrategias, probabilidades, rango_completo):
    """Crea una distribución completa con probabilidad 0 para estrategias no presentes"""
    min_val = rango_completo[0]
    max_val = rango_completo[-1]
    n = len(rango_completo)
    prob_completa = np.zeros(n)
    
    for i, estrategia in enumerate(estrategias):
        if estrategia < min_val or estrategia > max_val:
            continue  # Saltar estrategias fuera del rango
        idx = estrategia - min_val
        prob_completa[idx] = probabilidades[i]
    
    return prob_completa

def get_sort_value(s):
    """Función auxiliar para ordenar claves con valores numéricos o 'inf'"""
    parts = s.split('=')
    if len(parts) < 2:
        return float('inf')
    value_str = parts[1].strip()
    if value_str == 'inf':
        return float('inf')
    try:
        return float(value_str)
    except ValueError:
        return float('inf')

# Crear directorio para gráficos si no existe
graficos_dir = "graficos"
os.makedirs(graficos_dir, exist_ok=True)

# Extraer valores ordenados de λ y k
lambda_keys = sorted(data.keys(), key=get_sort_value)
k_keys = sorted(next(iter(data.values())).keys(), key=get_sort_value)

# Calcular rango de estrategias dinámicamente
all_strategies = []
for lambda_key in lambda_keys:
    for k_key in k_keys:
        agentes = data[lambda_key][k_key]
        for agente in agentes.values():
            if 'strategies' in agente:
                all_strategies.extend(agente['strategies'])
            elif 'estrategias' in agente:
                all_strategies.extend(agente['estrategias'])

if not all_strategies:
    print("❌ Error: No se encontraron estrategias en los datos")
    exit()

min_strategy = min(all_strategies)
max_strategy = max(all_strategies)
rango_estrategias = list(range(min_strategy, max_strategy + 1))
num_estrategias = len(rango_estrategias)

# Crear figura (ajustar tamaño según número de estrategias)
fig_width = max(10, len(lambda_keys) * 3)
fig_height = max(8, len(k_keys) * 2.5)
fig, axs = plt.subplots(len(k_keys), len(lambda_keys), 
                        figsize=(fig_width, fig_height), 
                        sharex=True, sharey=True)

# Manejar diferentes configuraciones de subplots
if len(k_keys) == 1 and len(lambda_keys) == 1:
    axs = [[axs]]
elif len(k_keys) == 1:
    axs = [axs]
elif len(lambda_keys) == 1:
    axs = [[ax] for ax in axs]
axs = np.array(axs)

# Diccionario para almacenar estadísticas
estadisticas = {}

print("\n=== ESTADÍSTICAS DE LAS DISTRIBUCIONES ===\n")

for row_idx, k_key in enumerate(k_keys):
    for col_idx, lambda_key in enumerate(lambda_keys):
        if len(k_keys) > 1 or len(lambda_keys) > 1:
            ax = axs[row_idx, col_idx]
        else:
            ax = axs[0, 0]

        agentes = data[lambda_key][k_key]
        agente = next(iter(agentes.values()))  # Tomamos el primer agente

        # Manejar diferentes nombres de claves
        if 'strategies' in agente:
            estrategias = agente['strategies']
            probabilidades = agente['probabilities']
        elif 'estrategias' in agente:
            estrategias = agente['estrategias']
            probabilidades = agente['probabilidades']
        else:
            print(f"❌ Error: No se encontraron las claves esperadas en {lambda_key}, {k_key}")
            continue

        # Crear distribución completa
        prob_completa = crear_distribucion_completa(estrategias, probabilidades, rango_estrategias)

        # Calcular estadísticas
        media, mediana, desviacion = calcular_estadisticas(estrategias, probabilidades)
        
        # Guardar estadísticas
        clave = f"{lambda_key}_{k_key}"
        estadisticas[clave] = {
            'media': float(media),
            'mediana': float(mediana),
            'desviacion': float(desviacion)
        }
        
        # Imprimir estadísticas
        print(f"{lambda_key}, {k_key}:")
        print(f"  Media: {media:.4f}")
        print(f"  Mediana: {mediana:.4f}")
        print(f"  Desviación estándar: {desviacion:.4f}")
        print()

        # Crear gráfico de barras
        bar_width = 0.8
        ax.bar(rango_estrategias, prob_completa, width=bar_width, color='steelblue', alpha=0.7)
        ax.set_ylim(0, max(prob_completa) * 1.15)
        
        # Resaltar media y mediana
        ax.axvline(x=media, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Media')
        ax.axvline(x=mediana, color='g', linestyle='-.', linewidth=1.5, alpha=0.7, label='Mediana')
        
        # Configurar etiquetas
        if col_idx == 0:
            ax.set_ylabel(f"{k_key}\n$P(a_i)$", fontsize=10)
        if row_idx == len(k_keys) - 1:
            ax.set_xlabel("Estrategia ($a_i$)", fontsize=10)

        if row_idx == 0:
            lambda_val = lambda_key.split('=')[1]
            ax.set_title(f"$\\lambda = {lambda_val}$", fontsize=12, pad=10)

        # Configurar ticks del eje x
        x_ticks_step = max(1, (max_strategy - min_strategy) // 10)
        ax.set_xticks(range(min_strategy, max_strategy + 1, x_ticks_step))
        
        # Añadir leyenda solo en el primer gráfico
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

# Ajustar espacios
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.2, wspace=0.15)

# Título general
fig.suptitle("Distribución de Probabilidades de Estrategias", fontsize=16, y=0.98)

# Guardar figura
output_path = os.path.join(graficos_dir, "distribucion_estrategias.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"\n✅ Gráfico guardado en: {output_path}")

# Guardar estadísticas
stats_file = os.path.join(graficos_dir, "estadisticas.json")
with open(stats_file, 'w') as f:
    json.dump(estadisticas, f, indent=2)
print(f"✅ Estadísticas guardadas en: {stats_file}")

# Tabla resumen
print("\n=== RESUMEN ESTADÍSTICO ===")
print(f"{'Parámetros':<25} {'Media':<10} {'Mediana':<10} {'Desv. Est.':<10}")
print("-" * 60)
for clave, stats in estadisticas.items():
    print(f"{clave:<25} {stats['media']:<10.4f} {stats['mediana']:<10.4f} {stats['desviacion']:<10.4f}")

print("\n✅ Proceso completado exitosamente!")