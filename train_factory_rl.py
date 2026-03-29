import numpy as np
import pandas as pd
import time
import os
from factory_gym_env import FactoryGymEnv

# Member 2: AI Engineer - Custom Lightweight Policy Gradient (as a PPO alternative)
# This bypasses the need for 'torch' and uses pure Numpy for stability on low-space devices.
class IndustrialRLAgent:
    def __init__(self, state_dim=13, action_dim=7):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Simple policy weights
        self.weights = np.random.randn(state_dim, action_dim) * 0.01

    def select_action(self, state):
        # Heuristic + Randomness for initial training
        action = np.dot(state, self.weights)
        # Add exploration noise
        action += np.random.normal(0, 0.1, size=self.action_dim)
        return np.clip(action, -1.0, 1.0)

    def train(self, state, action, reward, next_state):
        # Extremely simplified policy update for Day 7 demonstration
        # In a real PPO, this would involve the clipped surrogate objective
        learning_rate = 0.001
        gradient = np.outer(state, action) * reward
        self.weights += learning_rate * gradient

# Member 3: Reward Logger - Merged into FactoryLogger for Phase 3
# Legacy RewardLogger removed

# Main Training Script for Phase 3
def run_training(episodes=5):
    from data_pipeline import FactoryLogger
    env = FactoryGymEnv(render=False)
    agent = IndustrialRLAgent()
    logger = FactoryLogger(mode="ai")
    
    print(f"🚀 Starting RL Training Phase (Phase 3 AI Mode)...")
    for ep in range(episodes):
        obs, _ = env.reset()
        # [Day 10] Link simulation logger to ai mode
        env.sim.logger = logger
        
        total_reward = 0
        for step in range(100):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            agent.train(obs, action, reward, next_obs)
            
            # Member 3: Phase 3 Step Logging
            # We log placeholders for distance and idle time during training
            logger.log_step(step*0.1, 0.5, 0.1, "AI Training", reward=reward)
            
            obs = next_obs
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward:.2f}")
    
    logger.save_log()
    print("✅ Training complete. AI metrics saved.")

if __name__ == "__main__":
    run_training()
