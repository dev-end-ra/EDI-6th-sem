import os
import subprocess
import time

def run():
    print("🚀 Starting 15-Day Industrial Robot Project (Days 1-4)...")
    
    # 1. Run Industrial Simulation
    print("\n[Member 1 & 2] Running Factory Simulation (Conveyor + Robot + Stations)...")
    subprocess.run(["python3", "run_industrial_sim.py"])
    
    # 2. Check Logs
    if os.path.exists("factory_logs.csv"):
        print("\n[Member 3] Industrial Logs captured in 'factory_logs.csv'")
    
    # 3. Launch Dashboard
    print("\n[Member 4] Launching Factory Dashboard...")
    subprocess.run(["python3", "-m", "streamlit", "run", "app.py", "--browser.gatherUsageStats", "false"])

if __name__ == "__main__":
    run()
