import sys
sys.path.append('../')
import numpy as np
import time
from ch24 import IteratedBestResponse, NashEquilibrium, HierarchicalSoftmax
from problems.SimpleGames import TravelersDilemma


def print_separator(char="=", length=80):
    """Imprime una línea separadora"""
    print(char * length)


def print_section_header(title, char="="):
    """Imprime un encabezado de sección con formato"""
    print_separator(char)
    print(f" {title}")
    print_separator(char)


def identificar_juego(p, v):
    """Identifica y muestra información del juego"""
    juegos = {
        (5, 100): "Juego 1: Penalidad Baja, Valor Alto",
        (35, 100): "Juego 2: Penalidad Alta, Valor Alto", 
        (5, 70): "Juego 3: Penalidad Baja, Valor Medio",
        (35, 70): "Juego 4: Penalidad Alta, Valor Medio",
        (0, 70): "Juego 5: Sin Penalidad, Valor Medio",
        (0, 20): "Juego 6: Sin Penalidad, Valor Bajo"
    }
    
    if (p, v) in juegos:
        print_section_header(juegos[(p, v)])
        print(f"📊 Parámetros: Penalidad P = {p}, Valor Verdadero V = {v}")
        print()


def nash_equilibrium_TD(p=0, v=100, threshold=1e-2):
    """Calcula el equilibrio de Nash para el dilema del viajero"""
    Td = TravelersDilemma(P=p, V=v)
    N = NashEquilibrium()
    
    print(f"🎯 Calculando Equilibrio de Nash")
    print(f"   • Penalidad: {Td.P}")
    print(f"   • Valor Real: {Td.V}")
    print()
    
    policy = N.solve(Td)
    
    print("📋 Resultados del Equilibrio de Nash:")
    print_separator("-", 50)
    
    for i, policy_i in enumerate(policy):
        print(f"👤 Agente {i + 1}:")
        estrategias_significativas = {k: v for k, v in policy_i.p.items() if abs(v) > threshold}
        
        if estrategias_significativas:
            for k, v in estrategias_significativas.items():
                print(f"   • Estrategia {k}: {v:.4f} ({v*100:.2f}%)")
        else:
            print("   • No hay estrategias con probabilidad significativa")
        print()


def hierarchical_softmax_TD(p=0, v=100, lam=1.0, k=10, threshold=1e-4):
    """Ejecuta Hierarchical Softmax en el dilema del viajero"""
    Td = TravelersDilemma(P=p, V=v)
    
    print(f"🧠 Ejecutando Hierarchical Softmax")
    print(f"   • Penalidad: {Td.P}")
    print(f"   • Valor Real: {Td.V}")
    print(f"   • Parámetro de precisión (λ): {lam}")
    print(f"   • Niveles de iteración (k): {k}")
    print()
    
    HS = HierarchicalSoftmax.create_from_game(Td, lam=lam, k=k)
    policy = HS.solve(Td)
    
    print("📋 Resultados del Hierarchical Softmax:")
    print_separator("-", 50)
    
    for i, policy_i in enumerate(policy):
        print(f"👤 Agente {i + 1}:")
        estrategias_significativas = {k: v for k, v in policy_i.p.items() if abs(v) > threshold}
        
        if estrategias_significativas:
            # Ordenar por probabilidad descendente
            sorted_strategies = sorted(estrategias_significativas.items(), key=lambda x: x[1], reverse=True)
            for k, v in sorted_strategies:
                print(f"   • Estrategia {k}: {v:.4f} ({v*100:.2f}%)")
        else:
            print("   • No hay estrategias con probabilidad significativa")
        print()
def best_response(p=0, v=100):
    """Ejecuta la mejor respuesta iterada en el dilema del viajero"""
    Td = TravelersDilemma(P=p, V=v)
    
    print(f"🔄 Ejecutando Mejor Respuesta Iterada")
    print(f"   • Penalidad: {Td.P}")
    print(f"   • Valor Real: {Td.V}")
    print()
    
    M = IteratedBestResponse.create_from_game(Td, k_max=1)
    policy = M.solve(Td)
    
    print("📋 Resultados de la Mejor Respuesta Iterada:")
    print_separator("-", 50)
    
    for i, policy_i in enumerate(policy):
        print(f"👤 Agente {i + 1}:")
        for k, v in policy_i.p.items():
            if abs(v) > 1e-6:  # Solo mostrar probabilidades no nulas
                print(f"   • Estrategia {k}: {v:.4f} ({v*100:.2f}%)")
        print()


def format_time(seconds):
    """Formatea el tiempo de ejecución"""
    if seconds < 0.001:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 1:
        return f"{seconds*1000:.0f} ms"
    else:
        return f"{seconds:.2f} s"


def main():
    """Función principal"""
    print_section_header("🎮 ANÁLISIS DEL DILEMA DEL VIAJERO", "=")
    print("Comparación de métodos: Nash, Best Response y Hierarchical Softmax")
    print()
    
    time_start_total = time.time()
    juego_num = 0
    
    # Configuraciones de juegos a analizar
    configuraciones = [
        (5, 100), (35, 100), (5, 70), 
        (35, 70), (0, 70), (0, 20)
    ]
    
    # Parámetros para Hierarchical Softmax
    lambda_values = [0.5, 1.0, 2.0]  # Diferentes valores de precisión
    k_iterations = 10
    
    for p, v in configuraciones:
        juego_num += 1
        identificar_juego(p, v)
        
        # Equilibrio de Nash
        print("🎯 EQUILIBRIO DE NASH")
        print_separator("-", 40)
        time_start = time.time()
        try:
            nash_equilibrium_TD(p, v)
        except Exception as e:
            print(f"❌ Error calculando Nash: {e}")
        time_end = time.time()
        print(f"⏱️  Tiempo de ejecución: {format_time(time_end - time_start)}")
        print()
        
        # Mejor Respuesta Iterada
        print("🔄 MEJOR RESPUESTA ITERADA")
        print_separator("-", 40)
        time_start = time.time()
        try:
            best_response(p, v)
        except Exception as e:
            print(f"❌ Error calculando Best Response: {e}")
        time_end = time.time()
        print(f"⏱️  Tiempo de ejecución: {format_time(time_end - time_start)}")
        print()
        
        # Hierarchical Softmax con diferentes valores de lambda
        print("🧠 HIERARCHICAL SOFTMAX")
        print_separator("-", 40)
        
        for lam in lambda_values:
            print(f"🔸 Análisis con λ = {lam}")
            print_separator("·", 30)
            time_start = time.time()
            try:
                hierarchical_softmax_TD(p, v, lam=lam, k=k_iterations)
            except Exception as e:
                print(f"❌ Error calculando Hierarchical Softmax (λ={lam}): {e}")
            time_end = time.time()
            print(f"⏱️  Tiempo de ejecución: {format_time(time_end - time_start)}")
            print()
        
        if juego_num < len(configuraciones):
            print("\n" + "="*80 + "\n")
    
    time_end_total = time.time()
    
    print_section_header("📊 RESUMEN FINAL")
    print(f"✅ Análisis completado para {len(configuraciones)} configuraciones")
    print(f"🎯 Métodos comparados: Nash, Best Response, Hierarchical Softmax")
    print(f"🧠 Parámetros HS: λ = {lambda_values}, k = {k_iterations}")
    print(f"⏱️  Tiempo total de ejecución: {format_time(time_end_total - time_start_total)}")
    print_separator("=")


if __name__ == "__main__":
    main()




import sys; sys.path.append('../');

import os
import json
import numpy as np
import time
from ch24 import  IteratedBestResponse,NashEquilibrium, HierarchicalSoftmax
from problems.SimpleGames import TravelersDilemma

def identify_game(p, v):
    if p == 5 and v == 100:
        print("=======Game 1==========================================")
        print("game 1: Penalty P = 5, Real Value V = 100")
    elif p == 35 and v == 100:
        print("=======game 2==========================================")
        print("game 2: Penalty P = 35, Real Value V = 100")
    elif p == 5 and v == 70:
        print("=======game 3==========================================")
        print("game 3: Penalty P = 5, Real Value V = 70")
    elif p == 35 and v == 70:
        print("=======game 4==========================================")        
        print("game 4: Penalty P = 35, Real Value V = 70")
    elif p == 0 and v == 70:
        print("=======game 5==========================================")
        print("game 5: Penalty P = 0, Real Value V = 70")
    elif p == 0 and v == 20:
        print("=======game 6==========================================")
        print("game 6: Penalty P = 0, Real Value V = 20")


def identifier_game(p, v):
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

def nash_equilibrium_TD(p=0, v=100):
    Td = TravelersDilemma(P=p, V=v)
    N = NashEquilibrium()
    
    print(f"\n ERunning a Nash Equilibrium")
    print(f"   • Penalty: {Td.P}")
    print(f"   • Real Value: {Td.V}\n")

    policy = N.solve(Td)

    
    results = {
        "lambda=inf": {
            "k=inf": {}
        }
    }

    for i, policy_i in enumerate(policy):
        print(f"Agent {i}:")
        
        sorted_strategies = sorted(policy_i.p.items(), key=lambda x: x[1], reverse=True)
        for k, v_prob in sorted_strategies:
            print(f"   • Estrategy {k}: {v_prob:.4f} ({v_prob*100:.2f}%)")
        
        strategies, probabilities = zip(*sorted_strategies) if sorted_strategies else ([], [])

        results["lambda=inf"]["k=inf"][f"agent_{i}"] = {
            "strategies": list(strategies),
            "probabilities": list(probabilities)
        }

        print()


    output_dir = os.path.join(os.path.dirname(__file__), 'datos')
    os.makedirs(output_dir, exist_ok=True)


    game_id = identifier_game(p, v)
    filename = f"nash_equilibrium_game_{game_id}.json"
    output_path = os.path.join(output_dir, filename)


    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f" Complete policies saved in: {output_path}")

def hierarchical_softmax_TD(p, v, lam, k, all_results):
    Td = TravelersDilemma(P=p, V=v)

    print(f"Running a Hierarchical Softmax")
    print(f"   • Penalty: {Td.P}")
    print(f"   • Real Value: {Td.V}")
    print(f"   • Precision (λ): {lam}")
    print(f"   • Depths of rationality (k): {k}")
    print()

    HS = HierarchicalSoftmax.create_from_game(Td, lam=lam, k=k)
    policy = HS.solve(Td)

    print("Results of Hierarchical Softmax:")

    lambda_key = f"lambda={lam}"
    k_key = f"k={k}"
    
    if lambda_key not in all_results:
        all_results[lambda_key] = {}
    if k_key not in all_results[lambda_key]:
        all_results[lambda_key][k_key] = {}

    for i, policy_i in enumerate(policy):
        print(f"Agent {i + 1}:")

        sorted_strategies = sorted(policy_i.p.items(), key=lambda x: x[1], reverse=True)

        for estr, prob in sorted_strategies:
            print(f"   • Strategy {estr}: {prob:.4f} ({prob*100:.2f}%)")

        strategies, probabilities = zip(*sorted_strategies) if sorted_strategies else ([], [])

        all_results[lambda_key][k_key][f"agent_{i+1}"] = {
            "strategies": list(strategies),
            "probabilities": list(probabilities)
        }

        print()

def save_hierarchical_softmax_results(all_results, p, v):
    """Guarda todos los resultados de Hierarchical Softmax en un solo archivo JSON."""
    output_dir = os.path.join(os.path.dirname(__file__), 'datos')
    os.makedirs(output_dir, exist_ok=True)
    
    game_id = identifier_game(p, v)
    filename = f"hierarchical_softmax_game_{game_id}.json"
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"✅ Todas las políticas de Hierarchical Softmax guardadas en: {output_path}")


if __name__ == "__main__":

    #for p,v in zip([0],[100]):
    time_start_total = time.time()
    for v in [3,5]:  
        for p in [0]:
            if (p == 0 and v == 100) or (p == 5 and v == 20) or (p == 35 and v == 20):
                pass 
            else:

                identify_game(p, v)
                iden=identifier_game(p,v)
                print("=========================================================")
                print(f"Calculate the Nash Equilibrium for Traveler's Dilemma")
                time_start = time.time()
                print("Running...")
                nash_equilibrium_TD(p,v)
                time_end = time.time()
                print("Time taken: " + str((time_end - time_start)) + " s")
                print("\n")
                print("=========================================================")
                print("\n")
                print("Iterated Best Response on Traveler's Dilemma")
                time_start = time.time()
                print("Running...")
                #best_response(p,v)
                time_end = time.time()

                # Diccionario para acumular TODOS los resultados de Hierarchical Softmax
                hierarchical_results = {}
                    
                # Parámetros para Hierarchical Softmax
                lambda_values = [1,0.2]  # Diferentes valores de precisión
                k_iterations = [5,2]

                for lam in lambda_values:
                    for k_it in k_iterations:
                        print(f"🔸 Análisis con λ = {lam}")
                        print(f"🔸 Iteraciones k = {k_it}")
                        time_start = time.time()
                        try:
                            # Pasar el diccionario acumulativo
                            hierarchical_softmax_TD(p, v, lam=lam, k=k_it, all_results=hierarchical_results)
                        except Exception as e:
                            print(f"❌ Error calculando Hierarchical Softmax (λ={lam}): {e}")
                        time_end = time.time()
                        print(f"⏱️  Tiempo de ejecución: {time_end - time_start}")
                        print()
                
                # Guardar TODOS los resultados al final
                save_hierarchical_softmax_results(hierarchical_results, p, v)
                
    time_end_total = time.time()
    print("Time taken: " + str((time_end_total - time_start_total)) + " s")



    intento 

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
