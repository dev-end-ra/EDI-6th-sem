import pybullet as p
from factory_sim import FactorySimulation
import pandas as pd
import time

def run_production_cycle():
    print("🚀 Starting Industrial Production Cycle (Days 2-4)...")
    sim = FactorySimulation(render=True)
    
    # Run simulation for a complete cycle
    for i in range(1200):
        sim.step_simulation()
        
        # Periodic CSV save (Member 3)
        if i % 100 == 0:
            df = pd.DataFrame(sim.logs)
            df.to_csv("factory_logs.csv", index=False)
            
    print("✅ Cycle Complete. Logs saved to factory_logs.csv")
    p.disconnect()

if __name__ == "__main__":
    run_production_cycle()
