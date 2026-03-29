import pandas as pd
import matplotlib.pyplot as plt
import os

def create_visualizations():
    csv_file = "metrics.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # Load data
    df = pd.read_csv(csv_file)
    print("Loaded metrics:")
    print(df)

    # Parse values (remove ' sec', ' m', etc. and convert to float)
    def clean_val(val):
        if isinstance(val, str):
            # Split by space and take first part
            num_str = val.split(' ')[0]
            try:
                return float(num_str)
            except ValueError:
                return val
        return val

    df['CleanValue'] = df['Value'].apply(clean_val)

    # Extract metrics for plotting
    cycle_time = df[df['Metric'] == 'Cycle Time']['CleanValue'].values[0]
    path_length = df[df['Metric'] == 'Path Length']['CleanValue'].values[0]
    idle_time = df[df['Metric'] == 'Idle Time']['CleanValue'].values[0]
    task_count = df[df['Metric'] == 'Task Count']['CleanValue'].values[0]

    # 1. Cycle Time vs Idle Time (Bar Chart)
    plt.figure(figsize=(8, 6))
    plt.bar(['Total Cycle Time', 'Idle Time'], [cycle_time, idle_time], color=['blue', 'orange'])
    plt.ylabel('Time (seconds)')
    plt.title('Cycle Time Breakdown')
    plt.savefig('cycle_time_graph.png')
    print("Generated cycle_time_graph.png")

    # 2. Path Distance (Single Bar)
    plt.figure(figsize=(6, 6))
    plt.bar(['Robot Movement'], [path_length], color='green')
    plt.ylabel('Distance (meters)')
    plt.title('Total Path Length')
    plt.savefig('path_distance.png')
    print("Generated path_distance.png")

    # 3. Task Completion (Pie Chart/Status)
    plt.figure(figsize=(6, 6))
    plt.pie([task_count], labels=['Completed Tasks'], autopct='%1.1f%%', colors=['lightgreen'])
    plt.title('Task Completion Status')
    plt.savefig('task_completion.png')
    print("Generated task_completion.png")

    print("\nVisualization complete. All graphs saved as PNG files.")

if __name__ == "__main__":
    create_visualizations()
