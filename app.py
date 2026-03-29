import streamlit as st
import pandas as pd
import time
import os
from factory_sim import FactorySimulation

st.set_page_config(page_title="Factory Logistics Dashboard", layout="wide")

st.title("🏭 Industrial Workflow Dashboard (Days 2-4)")
st.markdown("### Real-time Monitoring: Conveyor, Robot & Stations")

# Member 4: Dashboard Logic
if 'sim_started' not in st.session_state:
    st.session_state.sim_started = False

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("System Status")
    if st.button("▶️ Start Simulation Phase"):
        st.session_state.sim_started = True
        
    if st.session_state.sim_started:
        st.success("Simulation Engine Active")
        # In a real app, this would be a background process
        # For Day 2-4 we show a live table from generated logs
    else:
        st.info("Simulation Idle")

with col2:
    st.subheader("Workpiece Metrics (Member 3)")
    if os.path.exists("factory_logs.csv"):
        df = pd.read_csv("factory_logs.csv")
        st.metric("Current Station", df['status'].iloc[-1])
        st.metric("Workpiece X-Pos", df['workpiece_x'].iloc[-1])
    else:
        st.warning("No live data found.")

st.divider()

# Layout for Day 3 & 4
col_left, col_mid, col_right = st.columns(3)
with col_left:
    st.info("📦 **Assembly Station**\n\nStatus: Active")
with col_mid:
    st.info("🎨 **Painting Station**\n\nStatus: Waiting")
with col_right:
    st.info("🔍 **Inspection Station**\n\nStatus: Waiting")

st.subheader("Live Simulation Logs")
if os.path.exists("factory_logs.csv"):
    st.dataframe(pd.read_csv("factory_logs.csv").tail(10), use_container_width=True)

if st.button("Refresh Results"):
    st.rerun()

st.sidebar.title("Day 1-4 Overview")
st.sidebar.markdown("""
- **Day 1**: Setup ✅
- **Day 2**: Conveyor ✅
- **Day 3**: Robot Arm ✅
- **Day 4**: Stations ✅
""")
