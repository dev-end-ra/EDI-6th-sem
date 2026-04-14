import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import csv
import time
import subprocess
import numpy as np
import pandas as pd

from factory_gym_env import FactoryGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# --- Prepare Directories ---
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)


class CustomLoggingCallback(BaseCallback):
    """
    Custom callback to log episode rewards to CSV and print mean reward every 5000 steps.
    """
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0
        self.reward_log_file = "data/reward_log.csv"
        self.print_freq = 5000
        
        # Initialize reward log file
        with open(self.reward_log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "step", "reward"])

    def _on_step(self):
        # Accumulate reward for the current episode
        # locals["rewards"] is an array because of DummyVecEnv
        self.current_reward += self.locals["rewards"][0]
        
        # Check if episode ended
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            with open(self.reward_log_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([len(self.episode_rewards), self.num_timesteps, round(self.current_reward, 2)])
            self.current_reward = 0
            
        # Print mean reward periodically
        if self.num_timesteps % self.print_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-20:])
                print(f"[Step {self.num_timesteps}] Mean Reward (last 20 eps): {mean_reward:.2f}")
            else:
                print(f"[Step {self.num_timesteps}] Mean Reward: N/A (no episodes completed yet)")
        return True


def collect_data(env, filename, num_episodes, policy_func, max_steps=6000):
    """
    Runs the environment for num_episodes and logs metrics to CSV.
    """
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "step", "cycle_time", "distance", "throughput", "idle_time", "task", "reward"])
        
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            step_count = 0
            while True:
                action = policy_func(obs, step_count)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Extract and write data exactly as requested
                writer.writerow([
                    ep + 1,
                    step_count + 1,
                    info.get("cycle_time", 0.0),
                    info.get("distance", 0.0),
                    info.get("throughput", 0),
                    info.get("idle_time", 0.0),
                    info.get("task", "Unknown"),
                    reward
                ])
                
                step_count += 1
                if terminated or truncated or step_count >= max_steps:
                    break
                    
            if ep % 10 == 0 or ep == num_episodes:
                print(f"Eval episode {ep}/{num_episodes} done")


def main():
    # Define our manual dummy action sequence for baseline
    dummy_actions = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    ]
    def manual_policy(obs, step):
        return dummy_actions[step % 5]

    print("=== Step 1: Collecting Baseline Data (50 episodes) ===")
    baseline_env = FactoryGymEnv(render=False, is_baseline=True)
    collect_data(baseline_env, "data/metrics_baseline.csv", num_episodes=50, policy_func=manual_policy)
    print("-> Baseline data saved to data/metrics_baseline.csv\n")
    
    print("=== Step 2: Training Clean PPO Agent (50,000 steps) ===")
    # Wrap environment with DummyVecEnv as required by stable-baselines3
    train_env = DummyVecEnv([lambda: FactoryGymEnv(render=False)])
    
    # Initialize PPO model with exactly specified hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        verbose=0
    )
    
    # Setup Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="models/checkpoints/", name_prefix="ppo_factory")
    logging_callback = CustomLoggingCallback()
    
    # Train the model
    # Note: On a mid-range laptop this takes 10 - 20 minutes for 50k steps.
    model.learn(total_timesteps=50000, callback=[checkpoint_callback, logging_callback])
    
    model.save("models/ppo_factory")
    print("-> Training complete. Model saved to models/ppo_factory.zip\n")
    
    print("=== Step 3: Evaluating Trained Model (50 episodes) ===")
    # Reload model just to be certain we are using the saved file properly
    loaded_model = PPO.load("models/ppo_factory")
    
    def ai_policy(obs, step):
        # deterministic=True evaluates the actual trained policy without exploration noise
        action, _states = loaded_model.predict(obs, deterministic=True)
        return action
        
    ai_env = FactoryGymEnv(render=False)
    collect_data(ai_env, "data/metrics_ai.csv", num_episodes=50, policy_func=ai_policy)
    print("-> AI metrics saved to data/metrics_ai.csv\n")
    
    print("=== Step 4: Comparing Results ===")
    df_base = pd.read_csv("data/metrics_baseline.csv")
    df_ai = pd.read_csv("data/metrics_ai.csv")
    
    # We want average cycle time and throughput. 
    # Since cycle_time is accumulated throughout an episode, the "final" cycle time of an episode 
    # is best captured by looking at the maximum cycle time per episode.
    # Same for throughput (cumulative doors completed).
    
    base_cycle = df_base.groupby('episode')['cycle_time'].max().mean()
    ai_cycle = df_ai.groupby('episode')['cycle_time'].max().mean()
    imp_cycle = ((ai_cycle - base_cycle) / base_cycle) * 100 if base_cycle != 0 else 0
    
    base_tp = df_base.groupby('episode')['throughput'].max().mean()
    ai_tp = df_ai.groupby('episode')['throughput'].max().mean()
    imp_tp = ((ai_tp - base_tp) / base_tp) * 100 if base_tp != 0 else 0
    
    print("========== RESULTS ==========")
    print(f"Baseline avg cycle time : {base_cycle:.2f} sec")
    print(f"AI avg cycle time       : {ai_cycle:.2f} sec")
    # For cycle time, reduction is good, so we use absolute value for a "positive" improvement phrasing if it went down.
    # The prompt explicitly asks for "Improvement: X.X%" structure
    # A negative cycle time percentage is actually an improvement. Let's frame it relative.
    print(f"Improvement             : {-imp_cycle:.2f}%\n")
    
    print(f"Baseline avg throughput : {base_tp:.2f}")
    print(f"AI avg throughput       : {ai_tp:.2f}")
    print(f"Improvement             : {imp_tp:.2f}%")
    print("=============================\n")
    
    print("=== Step 5: Generating Charts ===")
    # Ensure charts.py executes using the current environment
    subprocess.run(["python", "charts.py"])
    print("-> Job done. All charts regenerated.")

if __name__ == "__main__":
    main()
