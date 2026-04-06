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
            
    # Member 3: Phase 3 Save
    sim.logger.save_log()
    print("✅ Cycle Complete. Baseline logs saved.")
    
    # [Phase 4] Keep open until manually closed
    print("👀 Simulation persistent. Close the window to exit.")
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/240.)

if __name__ == "__main__":
    run_production_cycle()
