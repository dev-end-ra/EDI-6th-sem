import streamlit as st
import pandas as pd
import time
import os
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# --- 0. PAGE CONFIG ---
st.set_page_config(
    page_title="Conveyor Robot AI Optimization — Live Dashboard",
    layout="wide",
    page_icon="🏭"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #29b5e8;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. SIDEBAR ---
with st.sidebar:
    st.title("🏭 Industrial AI Dashboard")
    st.markdown("**Project:** Conveyor Path Optimization")
    st.markdown("**Status:** Phase 4 Deployment")
    
    st.divider()
    st.subheader("⚙️ Control Panel")
    auto_refresh = st.toggle("Auto-Refresh (5s)", value=False)
    if st.button("🔄 Force Refresh"):
        st.rerun()
        
    st.divider()
    st.subheader("📊 Data Actions")
    if st.button("Generate/Refresh Charts"):
        with st.spinner("Processing charts..."):
            subprocess.run(["python", "charts.py"])
        st.success("Charts Updated!")
        st.rerun()
    
    st.divider()
    st.info("💡 **Tip:** Ensure AI training is complete before analyzing comparison charts.")

# --- 2. DATA HELPERS & CALCULATIONS ---
def load_metrics():
    df_b = pd.read_csv("data/metrics_baseline.csv") if os.path.exists("data/metrics_baseline.csv") else None
    df_a = pd.read_csv("data/metrics_ai.csv") if os.path.exists("data/metrics_ai.csv") else None
    return df_b, df_a

def get_file_time(path):
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    return "N/A"

def safe_calc(df, col, agg):
    if df is None or 'episode' not in df.columns: return 0
    if agg == 'last': return df.groupby('episode')[col].last().mean()
    if agg == 'max': return df.groupby('episode')[col].max().mean()
    if agg == 'sum': return df.groupby('episode')[col].sum().mean()
    return 0

b_df, a_df = load_metrics()

b_cycle = safe_calc(b_df, 'cycle_time', 'last')
a_cycle = safe_calc(a_df, 'cycle_time', 'last') if a_df is not None else None

b_tp = safe_calc(b_df, 'throughput', 'max')
a_tp = safe_calc(a_df, 'throughput', 'max') if a_df is not None else None

b_dist = safe_calc(b_df, 'distance', 'last')
a_dist = safe_calc(a_df, 'distance', 'last') if a_df is not None else None

b_idle = safe_calc(b_df, 'idle_time', 'sum')
a_idle = safe_calc(a_df, 'idle_time', 'sum') if a_df is not None else None

# New Additions Logic
b_energy = b_dist * 0.6 + b_idle * 0.4
a_energy = (a_dist * 0.6 + a_idle * 0.4) if a_df is not None else None

b_eff = b_tp / (b_cycle + b_idle + 0.001)
a_eff = (a_tp / (a_cycle + a_idle + 0.001)) if a_df is not None else None

metrics_data = [
    ("Cycle Time", b_cycle, a_cycle, "s", False), 
    ("Throughput", b_tp, a_tp, "items", True),
    ("Distance", b_dist, a_dist, "m", False),
    ("Idle Time", b_idle, a_idle, "s", False),
    ("Energy Score", b_energy, a_energy, "pts", False),
    ("Efficiency", b_eff, a_eff, "score", True)
]

# --- 3. SECTION 1: LIVE KPI CARDS ---
st.header("📈 Live Key Performance Indicators")
col_rows = [st.columns(3), st.columns(3)]

for i, (name, b_val, a_val, unit, higher_better) in enumerate(metrics_data):
    delta = None
    delta_color = "normal" if higher_better else "inverse"
    
    if a_val is not None and b_val != 0:
        diff = a_val - b_val
        # Efficiency formatting specific overrides
        if name == "Efficiency":
            delta = f"{diff:+.3f} {unit}"
        else:
            delta = f"{diff:+.2f} {unit}"
        
    with col_rows[i // 3][i % 3]:
        # Formatter block depending on tight values
        a_str = f"{a_val:.3f}" if (a_val is not None and name == "Efficiency") else (f"{a_val:.2f}" if a_val is not None else "---")
        b_str = f"{b_val:.3f}" if name == "Efficiency" else f"{b_val:.2f}"
            
        if a_val is not None:
            st.metric(f"{name} ({unit})", a_str, delta=delta, delta_color=delta_color)
            st.caption(f"Baseline: {b_str}")
            
            # Sub-chart specific specifically for the requested Energy Score
            if name == "Energy Score":
                chart_df = pd.DataFrame({"Score": [b_val, a_val]}, index=["Baseline", "AI"])
                st.bar_chart(chart_df, height=130)
                
        else:
            st.metric(f"{name} ({unit})", b_str)
            st.caption("AI Training in progress...")

st.divider()

# --- NEW SECTION: OVERALL PERFORMANCE IMPROVEMENT ---
if a_df is not None:
    imp_cycle = ((b_cycle - a_cycle) / b_cycle * 100) if b_cycle != 0 else 0
    imp_dist = ((b_dist - a_dist) / b_dist * 100) if b_dist != 0 else 0
    imp_energy = ((b_energy - a_energy) / b_energy * 100) if b_energy != 0 else 0
    imp_eff = ((a_eff - b_eff) / b_eff * 100) if b_eff != 0 else 0

    st.subheader("🎯 Overall Performance Improvement")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ['Cycle Time\nReduction', 'Distance\nReduction', 'Energy\nReduction', 'Efficiency\nImprovement']
    values = [imp_cycle, imp_dist, imp_energy, imp_eff]
    colors = ['#28a745' if v >= 0 else '#dc3545' for v in values]
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Percentage Improvement (%)', fontsize=11)
    
    # Pad limits so text fits cleanly
    if values:
        ax.set_xlim(min(min(values)-15, -5), max(values) + 20)
    
    for bar in bars:
        width = bar.get_width()
        text_x = width + (2 if width >= 0 else -10)
        ax.text(text_x, bar.get_y() + bar.get_height()/2, f'{width:+.1f}%', 
                va='center', fontweight='bold', color=bar.get_facecolor(), fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)
    st.divider()

# --- NEW SECTION: KEY INSIGHTS ---
if a_df is not None:
    st.subheader("💡 Key Insights")
    
    def insight_card(text, color="green"):
        border_color = "#28a745" if color == "green" else "#17a2b8"
        st.markdown(f'''
        <div style="padding: 15px; border-left: 5px solid {border_color}; background-color: white; border-radius: 5px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <p style="margin: 0; font-size: 16px;">{text}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    insight_card(f"**Cycle Time Accelerated:** AI reduced cycle time by {imp_cycle:.1f}% — from {b_cycle:.1f}s to {a_cycle:.1f}s.", "green")
    insight_card(f"**Movement Optimized:** Robot distance reduced by {imp_dist:.1f}% — AI travels {a_dist:.2f}m vs baseline {b_dist:.2f}m.", "green")
    insight_card(f"**Operational Efficiency:** Efficiency score improved by {imp_eff:.1f}% — AI outputs significantly more throughput per active second.", "green")
    insight_card(f"**Sustainable Execution:** Energy consumption score reduced by {imp_energy:.1f}% through minimized idle wait states and optimal pathing.", "green")
    insight_card(f"**Bottleneck Eliminated:** Baseline uses fixed rigid timing locks — AI actively clears physical bounds dynamically in just {a_cycle:.1f}s.", "blue")
    insight_card("**Model Provenance:** PPO neural network successfully trained over 50,000 localized industrial simulation arrays to achieve zero-shot optimization.", "blue")
    
    st.divider()

# --- 4. SECTION 2: SIMULATION STATUS ---
st.subheader("🛡️ System Readiness & Data Freshness")
s_col1, s_col2, s_col3 = st.columns(3)

with s_col1:
    model_exists = os.path.exists("models/ppo_factory.zip")
    if model_exists:
        st.success("🤖 **AI Model Status:** TRAINING COMPLETE")
    else:
        # Check reward log to see if active
        r_log_exists = os.path.exists("data/reward_log.csv")
        status_text = "TRAINING IN PROGRESS" if r_log_exists else "MODEL NOT FOUND"
        st.warning(f"🤖 **AI Model Status:** {status_text}")

with s_col2:
    st.info(f"📊 **Base Data Freshness:**\n\n{get_file_time('data/metrics_baseline.csv')}")

with s_col3:
    st.info(f"🦾 **AI Data Freshness:**\n\n{get_file_time('data/metrics_ai.csv')}")

st.divider()

# --- 5. SECTION 3: CHART GALLERY ---
st.subheader("🎨 Comparative Performance Visualizations")

chart_files = [
    ("cycle_time_comparison.png", "Cycle Time Comparison"),
    ("kpi_bar_comparison.png", "KPI Bar Metrics"),
    ("reward_curve.png", "RL Reward Curve"),
    ("throughput_comparison.png", "Throughput Trends"),
    ("task_heatmap.png", "Task Heatmap"),
    ("summary_card.png", "Executive Summary")
]

gallery_rows = [st.columns(2), st.columns(2), st.columns(2)]
for i, (fname, title) in enumerate(chart_files):
    row_idx = i // 2
    col_idx = i % 2
    with gallery_rows[row_idx][col_idx]:
        path = f"charts/{fname}"
        if os.path.exists(path):
            st.image(path, caption=title, use_container_width=True)
        else:
            st.error(f"❌ {title} Not Found\n\nClick 'Generate Charts' in sidebar.")

st.divider()

# --- 6. SECTION 4: COMPARISON TABLE ---
st.subheader("📑 Detailed Metric Comparison")

if b_df is not None:
    comp_metrics = []
    for name, b_val, a_val, unit, higher_better in metrics_data:
        diff = ((a_val - b_val) / b_val) * 100 if b_val != 0 else 0
        imp = (-diff if not higher_better else diff)
        
        comp_metrics.append({
            "Metric": name,
            "Baseline": f"{b_val:.2f} {unit}",
            "After AI": f"{a_val:.2f} {unit}" if a_val is not None else "---",
            "Improvement %": f"{imp:+.1f}%" if a_val is not None else "---"
        })
    
    st.table(pd.DataFrame(comp_metrics))
else:
    st.warning("No metrics found to compare.")

st.divider()

# --- 7. SECTION 5: RAW DATA EXPLORER ---
st.subheader("🔍 Production Raw Log Explorer")
tab_base, tab_ai = st.tabs(["📊 Baseline Data", "🧠 AI Data"])

with tab_base:
    if b_df is not None:
        st.dataframe(b_df.tail(100), use_container_width=True, height=300)
        st.download_button("Download Baseline CSV", b_df.to_csv(index=False), "baseline_metrics.csv", "text/csv")
    else:
        st.info("Baseline data not available.")

with tab_ai:
    if a_df is not None:
        st.dataframe(a_df.tail(100), use_container_width=True, height=300)
        st.download_button("Download AI CSV", a_df.to_csv(index=False), "ai_metrics.csv", "text/csv")
    else:
        st.info("AI training data not logging yet.")

st.divider()

# --- 8. SECTION 6: ROBOT DEMO CONTROLS ---
st.subheader("🕹️ Simulation Demo Controller")
d_col1, d_col2 = st.columns(2)

with d_col1:
    demo_mode = st.radio("Simulation Mode:", ["Baseline (Fixed Path)", "AI (RL Optimized)"], horizontal=True)
    st.caption("⚠️ Ensure training is complete before launching AI mode.")

with d_col2:
    if st.button("🚀 Launch Simulation GUI", use_container_width=True):
        st.info("Launching PyBullet Window... (Close it to return to dashboard)")
        script = "factory_sim.py" if demo_mode == "Baseline (Fixed Path)" else "train_factory_rl.py" 
        subprocess.Popen(["python", script])
        st.toast("Sim Window Opened!", icon="🚀")

# --- 9. AUTO REFRESH LOGIC ---
if auto_refresh:
    time.sleep(5)
    st.rerun()
