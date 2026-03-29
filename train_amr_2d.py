import numpy as np
import pandas as pd
import time
import os
from simulation_core_2d import WarehouseSimulation2D

# Member 2: AI Engineer - Custom Q-Learning Agent
class QLearningAgent:
    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon
        self.model = None

    def choose_action(self, obs):
        amr_pos = obs[:2]
        target_pos = obs[2:]
        # Simple heuristic: move towards target
        direction = target_pos - amr_pos
        norm = np.linalg.norm(direction)
        if norm > 0:
            action = (direction / norm) + np.random.uniform(-0.1, 0.1, 2)
        else:
            action = np.zeros(2)
        return np.clip(action, -1, 1)

# Member 3: Data Collector
class MetricsCollector:
    def __init__(self, log_file="warehouse_metrics.csv"):
        self.log_file = log_file
        self.data = []

    def collect(self, step, obs, reward, dist):
        self.data.append({
            "step": step,
            "x": obs[0], "y": obs[1],
            "target_dist": dist,
            "reward": reward,
            "timestamp": time.time()
        })

    def save(self):
        pd.DataFrame(self.data).to_csv(self.log_file, index=False)
        print(f"Metrics saved to {self.log_file}")

# Execution
def run():
    sim = WarehouseSimulation2D()
    agent = QLearningAgent()
    collector = MetricsCollector()
    
    print("Running Warehouse Simulation...")
    for i in range(100):
        obs = sim.get_observation()
        action = agent.choose_action(obs)
        sim.step(action)
        
        dist = np.linalg.norm(sim.amr_pos - sim.target_pos)
        reward = -dist
        collector.collect(i, obs, reward, dist)
        
        if dist < 0.5:
            print("Target Reached!")
            break
            
    collector.save()

if __name__ == "__main__":
    run()
