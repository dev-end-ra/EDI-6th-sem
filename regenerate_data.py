import os
import subprocess
import numpy as np

from factory_gym_env import FactoryGymEnv
from train_factory_rl import collect_data

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    print("=== Re-generating BASELINE Data ===")
    
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

    try:
        baseline_env = FactoryGymEnv(render=False)
        collect_data(baseline_env, "data/metrics_baseline.csv", num_episodes=50, policy_func=manual_policy, max_steps=3000)
        print("-> Baseline metrics successfully regenerated.\n")
    except Exception as e:
        print(f"Error during baseline collection: {e}")

    print("=== Re-generating AI EVALUATION Data ===")
    # Offload to evaluate_model.py to handle isolation properly
    subprocess.run(["python", "evaluate_model.py"])

if __name__ == "__main__":
    main()
