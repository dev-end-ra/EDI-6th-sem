import streamlit as st
import pandas as pd
import time
import os
import subprocess
from datetime import datetime

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

# --- 2. DATA HELPERS ---
def load_metrics():
    df_b = pd.read_csv("data/metrics_baseline.csv") if os.path.exists("data/metrics_baseline.csv") else None
    df_a = pd.read_csv("data/metrics_ai.csv") if os.path.exists("data/metrics_ai.csv") else None
    return df_b, df_a

def get_file_time(path):
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    return "N/A"

b_df, a_df = load_metrics()

# --- 3. SECTION 1: LIVE KPI CARDS ---
st.header("📈 Live Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

metrics_to_show = [
    ("Cycle Time", "cycle_time", "s", False), # lower = better
    ("Throughput", "throughput", "items", True), # higher = better
    ("Distance", "distance", "m", False),
    ("Idle Time", "idle_time", "s", False)
]

for i, (name, col, unit, higher_better) in enumerate(metrics_to_show):
    # ALL four metrics (cycle_time, throughput, distance, idle_time) are cumulative within an episode
    # Therefore, taking the max() per episode gives the final value of that episode, then we mean across episodes
    b_val = b_df.groupby('episode')[col].max().mean() if b_df is not None and 'episode' in b_df.columns else 0
    a_val = a_df.groupby('episode')[col].max().mean() if a_df is not None and 'episode' in a_df.columns else None
    
    delta = None
    # Streamlit inherently colors "normal" mode where positive=green, negative=red.
    # And "inverse" mode where positive=red, negative=green (which we want for cycle time, distance, idle time).
    delta_color = "normal" if higher_better else "inverse"
    
    if a_val is not None and b_val != 0:
        # User requested formula: delta = ai_value - baseline_value
        diff = a_val - b_val
        delta = f"{diff:+.2f} {unit}"
        
    with [col1, col2, col3, col4][i]:
        if a_val is not None:
            st.metric(f"{name} ({unit})", f"{a_val:.2f}", delta=delta, delta_color=delta_color)
            st.caption(f"Baseline: {b_val:.2f}")
        else:
            st.metric(f"{name} ({unit})", f"{b_val:.2f}")
            st.caption("AI Training in progress...")

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
    for name, col, unit, higher_better in metrics_to_show:
        b_val = b_df.groupby('episode')[col].max().mean() if 'episode' in b_df.columns else 0
        a_val = a_df.groupby('episode')[col].max().mean() if a_df is not None and 'episode' in a_df.columns else 0
        
        diff = ((a_val - b_val) / b_val) * 100 if b_val != 0 else 0
        imp = (-diff if not higher_better else diff)
        
        comp_metrics.append({
            "Metric": name,
            "Baseline": f"{b_val:.2f} {unit}",
            "After AI": f"{a_val:.2f} {unit}" if a_df is not None else "---",
            "Improvement %": f"{imp:+.1f}%" if a_df is not None else "---"
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
        # Note: In a real demo, AI mode might launch a specific evaluation script.
        subprocess.Popen(["python", script])
        st.toast("Sim Window Opened!", icon="🚀")

# --- 9. AUTO REFRESH LOGIC ---
if auto_refresh:
    time.sleep(5)
    st.rerun()
