import sys; sys.path.append('../');

import os
import json
import numpy as np
import time
from datetime import datetime
from ch24 import IteratedBestResponse, NashEquilibrium
from problems.SimpleGames import TravelersDilemma


class SimpleLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


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

    output_dir = os.path.join(os.path.dirname(__file__), 'datos_nash')
    os.makedirs(output_dir, exist_ok=True)

    game_id = identifier_game(p, v)
    filename = f"nash_equilibrium_game_{game_id}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"Complete policies saved in: {output_path}")


if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"execution_log_{timestamp}.txt"
    
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    

    logger = SimpleLogger(log_path)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print(f"=== EXECUTION STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"Log will be saved to: {log_path}")
        print("=" * 60)
        
        time_start_total = time.time()
        
        for v in [100, 70, 20]:  
            for p in [0, 5, 35]:
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
                    print("=========================================================")
                    print()
                    
        time_end_total = time.time()
        print("Total time taken: " + str((time_end_total - time_start_total)) + " s")
        print("=" * 60)
        print(f"=== EXECUTION FINISHED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\n📝 Execution log saved to: {log_path}")