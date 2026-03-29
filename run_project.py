import os
import subprocess
import time

def run():
    print("🚀 Starting 4-Member Warehouse AMR Project...")
    
    # 1. Run Simulation
    print("\n[Member 1 & 2] Running Simulation & AI Agent...")
    subprocess.run(["python3", "train_amr_2d.py"])
    
    # 2. Check Metrics
    if os.path.exists("warehouse_metrics.csv"):
        print("\n[Member 3] Metrics Collected successfully in 'warehouse_metrics.csv'")
    
    # 3. Launch Dashboard
    print("\n[Member 4] Launching Streamlit Dashboard...")
    print("Note: If prompted for an email, just press ENTER.")
    subprocess.run(["python3", "-m", "streamlit", "run", "dashboard.py", "--browser.gatherUsageStats", "false"])

if __name__ == "__main__":
    run()
