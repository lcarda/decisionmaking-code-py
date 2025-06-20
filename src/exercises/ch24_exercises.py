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
    
    print(f"\n Running a Nash Equilibrium")
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

    print(f"Complete policies saved in: {output_path}")

def hierarchical_softmax_TD(p, v, lam, k, all_results):
    """
    Función modificada que agrega resultados a la estructura all_results
    en lugar de guardar archivos separados
    """
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

    # Crear las claves si no existen
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

def save_all_hierarchical_results(p, v, all_results):
    """
    Guarda todos los resultados de hierarchical softmax en un solo archivo JSON
    """
    output_dir = os.path.join(os.path.dirname(__file__), 'datos')
    os.makedirs(output_dir, exist_ok=True)

    game_id = identifier_game(p, v)
    filename = f"hierarchical_softmax_game_{game_id}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    print(f"All Hierarchical Softmax results saved in: {output_path}")

if __name__ == "__main__":
    time_start_total = time.time()
    
    for v in [100,70,20]:  
        for p in [0,5,35]:
            if (p == 0 and v == 100) or (p == 5 and v == 20) or (p == 35 and v == 20):
                pass 
            else:
                identify_game(p, v)
                iden = identifier_game(p, v)
                print("=========================================================")
                print(f"Calculate the Nash Equilibrium for Traveler's Dilemma")
                time_start = time.time()
                print("Running...")
                nash_equilibrium_TD(p, v)
                time_end = time.time()
                print("Time taken: " + str((time_end - time_start)) + " s")
                print("\n")
                print("=========================================================")
                print("\n")
                   
                # Estructura para acumular todos los resultados de Hierarchical Softmax
                all_hierarchical_results = {}
                
                # Parámetros para Hierarchical Softmax
                lambda_values = [0, 0.5,1,3]  
                k_iterations = [0, 5,10,20]

                for lam in lambda_values:
                    for k_it in k_iterations:
                        time_start = time.time()
                        try:
                            hierarchical_softmax_TD(p, v, lam=lam, k=k_it, all_results=all_hierarchical_results)
                        except Exception as e:
                            print(f"❌ Error calculando Hierarchical Softmax (λ={lam}): {e}")
                        time_end = time.time()
                        print(f"\n ⏱️  Time taken: {time_end - time_start}")
                        print()
                
                # Guardar todos los resultados de Hierarchical Softmax en un solo archivo
                save_all_hierarchical_results(p, v, all_hierarchical_results)
                print("=========================================================")
                print()
                
    time_end_total = time.time()
    print("Total time taken: " + str((time_end_total - time_start_total)) + " s")