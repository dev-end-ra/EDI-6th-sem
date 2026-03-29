import numpy as np
import pandas as pd
import time
import os
from warehouse_gym_env import WarehouseGymEnv

# Member 2: AI Engineer - Custom Lightweight Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_dim=5, action_dim=2, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        # Discretize state and action for simple Q-Learning
        self.q_table = {} 

    def get_state_key(self, obs):
        # Discretize: [x, y, tx, ty, yaw] -> buckets
        return tuple(np.round(obs, 1))

    def choose_action(self, obs):
        state = self.get_state_key(obs)
        if np.random.uniform(0, 1) < self.epsilon:
            # Random exploration
            return np.random.uniform(-1, 1, 2)
        
        if state not in self.q_table:
            return np.zeros(2)
        
        return self.q_table[state]

    def learn(self, obs, action, reward, next_obs):
        state = self.get_state_key(obs)
        next_state = self.get_state_key(next_obs)
        
        if state not in self.q_table: self.q_table[state] = np.zeros(2)
        if next_state not in self.q_table: self.q_table[next_state] = np.zeros(2)
        
        # Simple Q-Update (adapted for continuous actions via simple mean or similar)
        # For simplicity in this demo, we'll just store the best action seen so far for each state
        if reward > -1: # Some threshold
            self.q_table[state] = action

# Member 3: Data Collector
class MetricsCollector:
    def __init__(self, log_file="warehouse_metrics.csv"):
        self.log_file = log_file
        self.data = []

    def collect(self, step, obs, reward, dist):
        entry = {
            "step": step,
            "x": obs[0],
            "y": obs[1],
            "target_dist": dist,
            "reward": reward,
            "timestamp": time.time()
        }
        self.data.append(entry)

    def save(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.log_file, index=False)
        print(f"Metrics saved to {self.log_file}")

# Main execution
def run_simulation(steps=1000, train=True):
    env = WarehouseGymEnv(render=True)
    agent = QLearningAgent()
    collector = MetricsCollector()
    
    obs, _ = env.reset()
    print("Starting AMR Navigation Simulation...")
    
    for i in range(steps):
        action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        if train:
            agent.learn(obs, action, reward, next_obs)
        
        dist = np.linalg.norm(next_obs[:2] - next_obs[2:4])
        collector.collect(i, obs, reward, dist)
        
        obs = next_obs
        if terminated or truncated:
            print("Target Reached or Simulation Ended.")
            break
            
    collector.save()
    env.close()

if __name__ == "__main__":
    run_simulation()
