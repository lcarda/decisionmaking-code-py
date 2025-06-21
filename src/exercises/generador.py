import numpy as np
import json

# Configurar semilla para reproducibilidad
np.random.seed(42)

# Parámetros de simulación
n_samples = 100

# Datos originales (solo para estructura de referencia)
original_data = {
    "lambda=1": {
        "k=0": {"agent_1": {}, "agent_2": {}},
        "k=2": {"agent_1": {}, "agent_2": {}}
    },
    "lambda=0.2": {
        "k=0": {"agent_1": {}, "agent_2": {}},
        "k=2": {"agent_1": {}, "agent_2": {}}
    }
}

# Datos simulados
simulated_data = {}

# Generar datos para cada combinación de lambda y k
for lambda_key in original_data.keys():
    lambda_val = float(lambda_key.split('=')[1])
    simulated_data[lambda_key] = {}
    
    for k_key in original_data[lambda_key].keys():
        k_val = int(k_key.split('=')[1])
        simulated_data[lambda_key][k_key] = {}
        
        # Parámetros para la distribución normal
        # k como media, lambda + 10 como desviación estándar
        mean = k_val
        std_dev = lambda_val + 10
        
        # Generar datos para agent_1 y agent_2
        for agent in ["agent_1", "agent_2"]:
            # Generar 100 muestras de distribución normal
            samples = np.random.normal(mean, std_dev, n_samples)
            
            # Redondear a enteros y asegurar que sean positivos
            samples = np.maximum(0, np.round(samples).astype(int))
            
            # Obtener estrategias únicas y sus frecuencias
            unique_strategies, counts = np.unique(samples, return_counts=True)
            
            # Convertir frecuencias a probabilidades
            probabilities = counts / n_samples
            
            # Guardar en el diccionario
            simulated_data[lambda_key][k_key][agent] = {
                "strategies": unique_strategies.tolist(),
                "probabilities": probabilities.tolist()
            }

# Convertir a JSON y mostrar
json_output = json.dumps(simulated_data, indent=4)
print(json_output)

# Opcional: guardar en archivo
with open('simulated_hierarchical_softmax_game_0_3.json', 'w') as f:
    json.dump(simulated_data, f, indent=4)

print(f"\n✅ Datos simulados generados con {n_samples} muestras por agente")
print("📊 Parámetros utilizados:")
print("   - Media: valor de k")
print("   - Desviación estándar: valor de lambda + 10")
print("   - Distribución: Normal truncada (valores >= 0)")