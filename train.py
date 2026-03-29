from stable_baselines3 import PPO
from factory_gym_env import FactoryGymEnv
import os

def train():
    # Create Environment (Headless)
    env = FactoryGymEnv(render=False)
    
    # Initialize PPO Agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_factory_logs/")
    
    print("Starting RL training...")
    model.learn(total_timesteps=100000)
    
    # Save Model
    model.save("ppo_factory_optimizer")
    print("Model saved to ppo_factory_optimizer.zip")

if __name__ == "__main__":
    train()
