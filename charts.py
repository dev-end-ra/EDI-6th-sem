import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Global Styling
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_data():
    paths = {
        "baseline": "data/metrics_baseline.csv",
        "ai": "data/metrics_ai.csv",
        "reward": "data/reward_log.csv"
    }
    
    data = {}
    for key, path in paths.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = None
    return data["baseline"], data["ai"], data["reward"]

def setup_plot(title, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    return fig, ax

def finalize_plot(fig, ax, filename):
    ax.text(0.99, 0.01, "Conveyor Robot AI Project", transform=ax.transAxes, 
            ha='right', va='bottom', fontsize=8, color='gray', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if not os.path.exists('charts'):
        os.makedirs('charts')
    fig.savefig(f"charts/{filename}", dpi=150)
    plt.close(fig)

def chart_1_cycle_time(df_base, df_ai):
    fig, ax = setup_plot("Cycle Time Before vs After AI")
    
    if df_base is not None:
        y_base = df_base['cycle_time'].rolling(window=10).mean()
        ax.plot(y_base, label='Baseline', color='red', alpha=0.7)
        ax.axhline(df_base['cycle_time'].mean(), color='red', linestyle='--', alpha=0.5, label=f'Base Mean: {df_base["cycle_time"].mean():.2f}s')
    
    if df_ai is not None:
        y_ai = df_ai['cycle_time'].rolling(window=10).mean()
        ax.plot(y_ai, label='AI', color='green', alpha=0.7)
        ax.axhline(df_ai['cycle_time'].mean(), color='green', linestyle='--', alpha=0.5, label=f'AI Mean: {df_ai["cycle_time"].mean():.2f}s')
    else:
        ax.text(0.5, 0.5, "AI data not available yet", transform=ax.transAxes, ha='center', color='gray')

    ax.set_xlabel("Step Number")
    ax.set_ylabel("Cycle Time (seconds)")
    ax.legend(frameon=False)
    finalize_plot(fig, ax, "cycle_time_comparison.png")

def chart_2_kpi(df_base, df_ai):
    fig, ax = setup_plot("Key Performance Indicators — Before vs After AI", figsize=(10, 6))
    
    metrics = ['cycle_time', 'distance', 'idle_time', 'throughput']
    labels = ['Avg Cycle Time (s)', 'Avg Distance (m)', 'Avg Idle Time (s)', 'Avg Throughput (u/h)']
    
    base_vals = [df_base[m].mean() for m in metrics] if df_base is not None else [0]*4
    ai_vals = [df_ai[m].mean() for m in metrics] if df_ai is not None else [0]*4
    
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline', color='red', alpha=0.6)
    rects2 = ax.bar(x + width/2, ai_vals, width, label='AI', color='green', alpha=0.6)
    
    ax.set_xticks(x)
    tick_labels = []
    for i, m in enumerate(metrics):
        if df_base is not None and df_ai is not None and base_vals[i] != 0:
            imp = ((ai_vals[i] - base_vals[i]) / base_vals[i]) * 100
            if m == 'throughput': # Higher is better
                label = f"{labels[i]}\n({imp:+.1f}%)"
            else: # Lower is better
                label = f"{labels[i]}\n({-imp:+.1f}% Imp.)"
        else:
            label = labels[i]
        tick_labels.append(label)
    
    ax.set_xticklabels(tick_labels)
    ax.legend(frameon=False)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    finalize_plot(fig, ax, "kpi_bar_comparison.png")

def chart_3_reward(df_reward):
    fig, ax = setup_plot("RL Training Reward Curve")
    
    if df_reward is not None:
        reward = df_reward['reward']
        ax.plot(reward, color='lightgray', alpha=0.5, linewidth=0.5, label='Raw Reward')
        rolling = reward.rolling(window=20).mean()
        ax.plot(rolling, color='blue', linewidth=2, label='Rolling Average (20)')
        
        # Mark consistently positive (e.g., last 10 points all > 0)
        pos_indices = np.where(rolling > 0)[0]
        if len(pos_indices) > 0:
            first_pos = pos_indices[0]
            ax.annotate('Reward Gains Positive', xy=(first_pos, rolling[first_pos]), 
                        xytext=(first_pos + 10, rolling[first_pos] + 5),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    else:
        ax.text(0.5, 0.5, "Reward log not available yet", transform=ax.transAxes, ha='center', color='gray')

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward Value")
    ax.legend(frameon=False)
    finalize_plot(fig, ax, "reward_curve.png")

def chart_4_heatmap(df_base, df_ai):
    fig, ax = setup_plot("Average Cycle Time per Task")
    
    tasks = ['Assembly', 'Painting', 'Inspection', 'Moving on Conveyor']
    data = []
    
    for df in [df_base, df_ai]:
        row = []
        if df is not None:
            for task in tasks:
                val = df[df['task'] == task]['cycle_time'].mean()
                row.append(val if not np.isnan(val) else 0)
        else:
            row = [0] * len(tasks)
        data.append(row)
    
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlOrRd", 
                xticklabels=['Assembly', 'Paint', 'Inspection', 'Output/Conv'],
                yticklabels=['Baseline', 'After AI'], ax=ax, cbar_kws={'label': 'Seconds'})
    
    finalize_plot(fig, ax, "task_heatmap.png")

def chart_5_throughput(df_base, df_ai):
    fig, ax = setup_plot("Throughput Comparison Over Episodes")
    
    # Simulate episodes by grouping indices into blocks of 50 steps
    def get_episodes(df):
        if df is None: return None
        if 'episode' in df.columns:
            return df.groupby('episode')['throughput'].max() # Use max per episode as our new throughput logic accumulates
        else:
            df = df.copy()
            df['episode'] = df.index // 50
            return df.groupby('episode')['throughput'].mean()

    base_ep = get_episodes(df_base)
    ai_ep = get_episodes(df_ai)
    
    if base_ep is not None:
        ax.plot(base_ep.index, base_ep.values, label='Baseline', color='red', alpha=0.8)
    
    if ai_ep is not None:
        ax.plot(ai_ep.index, ai_ep.values, label='AI', color='green', alpha=0.8)
        
        if base_ep is not None:
            # Match lengths for filling
            common_idx = ai_ep.index.intersection(base_ep.index)
            ax.fill_between(common_idx, base_ep[common_idx], ai_ep[common_idx], 
                            where=(ai_ep[common_idx] >= base_ep[common_idx]), 
                            color='lightgreen', alpha=0.3, label='AI Performance Gain')
    else:
        ax.text(0.5, 0.5, "AI data not available yet", transform=ax.transAxes, ha='center', color='gray')

    ax.set_xlabel("Episode Number (Grouped Steps)")
    ax.set_ylabel("Throughput (Units/Hr)")
    ax.legend(frameon=False)
    finalize_plot(fig, ax, "throughput_comparison.png")

def chart_6_summary(df_base, df_ai):
    fig, ax = setup_plot("Summary Metrics — Before vs After AI", figsize=(10, 4))
    ax.axis('off')
    
    if df_base is not None and df_ai is not None:
        metrics = {
            'Cycle Time': ('cycle_time', False), # False = reduction is good
            'Throughput': ('throughput', True),   # True = increase is good
            'Distance': ('distance', False),
            'Idle Time': ('idle_time', False)
        }
        
        for i, (name, (col, higher_better)) in enumerate(metrics.items()):
            base_val = df_base[col].mean()
            ai_val = df_ai[col].mean()
            
            diff = ((ai_val - base_val) / base_val) * 100 if base_val != 0 else 0
            
            # Improvement status
            is_improvement = (diff > 0 if higher_better else diff < 0)
            color = 'green' if is_improvement else 'red'
            arrow = "▲" if diff > 0 else "▼"
            
            # Display positions
            x_pos = 0.1 + i * 0.25
            ax.text(x_pos, 0.8, name, ha='center', fontsize=12, fontweight='bold')
            ax.text(x_pos, 0.5, f"{abs(diff):.1f}%", ha='center', fontsize=24, fontweight='bold', color=color)
            ax.text(x_pos, 0.3, arrow, ha='center', fontsize=20, color=color)
            
    else:
        ax.text(0.5, 0.5, "Comparison Summary (Wait for AI metrics...)", ha='center', fontsize=14, color='gray')

    finalize_plot(fig, ax, "summary_card.png")

def generate_all_charts():
    print("Starting Chart Generation Pipeline...")
    df_base, df_ai, df_reward = load_data()
    
    chart_1_cycle_time(df_base, df_ai)
    chart_2_kpi(df_base, df_ai)
    chart_3_reward(df_reward)
    chart_4_heatmap(df_base, df_ai)
    chart_5_throughput(df_base, df_ai)
    chart_6_summary(df_base, df_ai)
    
    print("All 6 charts generated in /charts folder.")

if __name__ == "__main__":
    generate_all_charts()
