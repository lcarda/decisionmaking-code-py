import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# Parámetros
p = 0
v = 3

def identificador_juego(p, v):
    if p == 5 and v == 100:
        return 1
    elif p == 35 and v == 100:
        return 2
    elif p == 5 and v == 70:
        return 3
    elif p == 35 and v == 70:
        return 4
    elif p == 0 and v == 70:
        return 5
    elif p == 0 and v == 20:
        return 6
    else:
        return f"{p}_{v}"

game_id = identificador_juego(p, v)

# Rutas
datos_dir = "/home/pop/Documents/Carrera/tomacode/decisionmaking-code-py/src/exercises/datos"
graficos_dir = "/home/pop/Documents/Carrera/tomacode/decisionmaking-code-py/src/exercises/graficos"
input_file = os.path.join(datos_dir, f"hierarchical_softmax_game_{game_id}.json")

os.makedirs(graficos_dir, exist_ok=True)

# Cargar datos
with open(input_file, 'r') as f:
    data = json.load(f)

# Extraer valores ordenados de λ y k
lambda_keys = sorted(data.keys(), key=lambda x: float(x.split('=')[1]))
k_keys = sorted(next(iter(data.values())).keys(), key=lambda x: int(x.split('=')[1]))

# Crear figura
fig, axs = plt.subplots(len(k_keys), len(lambda_keys), figsize=(10, 10), sharex=True, sharey=True)

if len(k_keys) == 1:
    axs = [axs]
if len(lambda_keys) == 1:
    axs = [[ax] for ax in axs]

for row_idx, k_key in enumerate(k_keys):
    for col_idx, lambda_key in enumerate(lambda_keys):
        ax = axs[row_idx][col_idx]

        agentes = data[lambda_key][k_key]
        agente = next(iter(agentes.values()))  # Tomamos el primero (p.ej., agent_1)

        estrategias = agente['estrategias']
        probabilidades = agente['probabilidades']

        ax.bar(estrategias, probabilidades, width=1.0)
        ax.set_ylim(0, 0.15)

        # Etiquetas solo en la izquierda y abajo
        if col_idx == 0:
            ax.set_ylabel(f"{k_key}", fontsize=12)
        if row_idx == len(k_keys) - 1:
            ax.set_xlabel("ai", fontsize=12)

        if row_idx == 0:
            ax.set_title(f"{lambda_key}", fontsize=12)

# Etiquetas generales
fig.text(0.04, 0.5, r'$P(a_i)$', va='center', rotation='vertical', fontsize=14)
fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])

# Guardar figura
output_path = os.path.join(graficos_dir, f"rejilla_game_{game_id}.png")
plt.savefig(output_path, bbox_inches='tight')
print(f"✅ Gráfico tipo rejilla guardado en: {output_path}")
