import os
import csv
import subprocess
import numpy as np
import pandas as pd
import pybullet as p

from stable_baselines3 import PPO
from factory_gym_env import FactoryGymEnv

def evaluate_and_collect(env, model_path, filename, num_episodes=50, max_steps=3000):
    # Ensure PyBullet is cleanly connected and previous instances disconnected
    if p.isConnected():
        p.disconnect()
        
    loaded_model = PPO.load(model_path)
    
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "step", "cycle_time", "distance", "throughput", "idle_time", "task", "reward"])
        
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            step_count = 0
            while True:
                # Deterministic=True evaluates the actual trained policy without exploration noise
                action, _states = loaded_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Extract and write data exactly as requested
                writer.writerow([
                    ep,
                    step_count + 1,
                    info.get("cycle_time", 0.0),
                    info.get("distance", 0.0),
                    info.get("throughput", 0),
                    info.get("idle_time", 0.0),
                    info.get("task", "Unknown"),
                    reward
                ])
                
                step_count += 1
                
                # Hard timeout if model gets stuck
                if terminated or truncated or step_count >= max_steps:
                    break
            
            # Print progress
            if ep % 10 == 0 or ep == num_episodes:
                print(f"Eval episode {ep}/{num_episodes} done")

def main():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    print("=== Standalone Evaluation: Assessing Trained Model ===")
    ai_env = FactoryGymEnv(render=False) # MUST BE p.DIRECT
    
    evaluate_and_collect(
        env=ai_env, 
        model_path="models/ppo_factory", 
        filename="data/metrics_ai.csv", 
        num_episodes=50, 
        max_steps=3000
    )
    
    print("-> AI metrics saved to data/metrics_ai.csv\n")
    
    print("=== Comparing Results ===")
    try:
        df_base = pd.read_csv("data/metrics_baseline.csv")
        base_cycle = df_base.groupby('episode')['cycle_time'].max().mean()
        base_tp = df_base.groupby('episode')['throughput'].max().mean()
    except Exception:
        base_cycle = 0
        base_tp = 0
        
    df_ai = pd.read_csv("data/metrics_ai.csv")
    ai_cycle = df_ai.groupby('episode')['cycle_time'].max().mean()
    ai_tp = df_ai.groupby('episode')['throughput'].max().mean()
    
    imp_cycle = ((ai_cycle - base_cycle) / base_cycle) * 100 if base_cycle != 0 else 0
    imp_tp = ((ai_tp - base_tp) / base_tp) * 100 if base_tp != 0 else 0
    
    print("========== RESULTS ==========")
    print(f"Baseline avg cycle time : {base_cycle:.2f} sec")
    print(f"AI avg cycle time       : {ai_cycle:.2f} sec")
    print(f"Improvement             : {-imp_cycle:.2f}%\n")
    
    print(f"Baseline avg throughput : {base_tp:.2f}")
    print(f"AI avg throughput       : {ai_tp:.2f}")
    print(f"Improvement             : {imp_tp:.2f}%")
    print("=============================\n")
    
    print("=== Generating Charts ===")
    subprocess.run(["python", "charts.py"])
    print("-> Job done. All charts regenerated.")

if __name__ == "__main__":
    main()
