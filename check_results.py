import pandas as pd

b = pd.read_csv("data/metrics_baseline.csv")
a = pd.read_csv("data/metrics_ai.csv")

print("Baseline cycle time:", b["cycle_time"].mean())
print("AI cycle time:", a["cycle_time"].mean())