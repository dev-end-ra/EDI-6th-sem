import os
import subprocess
import time

def run():
    print("🚀 Starting 15-Day Industrial Robot Project (Days 1-7)...")
    
    # 1. Run Industrial Simulation (Baseline)
    print("\n[Member 1 & 2] Running Factory Simulation (Baseline Cycle)...")
    subprocess.run(["python3", "run_industrial_sim.py"])
    
    # 2. Run RL Training (Member 2)
    print("\n[Member 2] Running AI Training (Custom RL)...")
    subprocess.run(["python3", "train_factory_rl.py"])
    
    # 3. Launch Dashboard
    print("\n[Member 4] Launching Optimized Factory Dashboard...")
    subprocess.run(["python3", "-m", "streamlit", "run", "app.py", "--browser.gatherUsageStats", "false"])

if __name__ == "__main__":
    run()
