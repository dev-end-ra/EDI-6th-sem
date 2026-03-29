import pandas as pd
import time
import os

class FactoryLogger:
    def __init__(self, mode="baseline"):
        self.mode = mode # "baseline" or "ai"
        self.filename = f"{mode}_metrics.csv"
        self.data = []
        
        # Consistent format for Phase 3
        self.columns = [
            'timestamp', 'cycle_time', 'throughput', 
            'robot_distance', 'idle_time', 'reward', 'status'
        ]

    def log_step(self, cycle_time, robot_dist, idle_time, status, reward=None):
        entry = {
            "timestamp": time.time(),
            "cycle_time": round(cycle_time, 2), # seconds
            "throughput": round(3600 / (cycle_time + 1e-6), 1), # units/hr
            "robot_distance": round(robot_dist, 3), # meters
            "idle_time": round(idle_time, 2), # seconds
            "reward": round(reward, 2) if reward is not None else "N/A",
            "status": status
        }
        self.data.append(entry)

    def save_log(self):
        df = pd.DataFrame(self.data)
        # Append mode for continuous storage (Day 11 requirement)
        if not os.path.isfile(self.filename):
            df.to_csv(self.filename, index=False)
        else:
            df.to_csv(self.filename, mode='a', header=False, index=False)
        
        print(f"📊 {self.mode.capitalize()} metrics saved to {self.filename}")

    def get_aggregated_metrics(self):
        if not self.data: return {}
        df = pd.DataFrame(self.data)
        return {
            "avg_cycle": df['cycle_time'].mean(),
            "total_dist": df['robot_distance'].sum(),
            "max_throughput": df['throughput'].max()
        }
