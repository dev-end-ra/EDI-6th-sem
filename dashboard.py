import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

st.set_page_config(page_title="Warehouse AMR Dashboard", layout="wide")

st.title("🚀 Warehouse AMR Logistics Dashboard")
st.markdown("### Real-time Monitoring & Resource Optimization")

# Member 4: UI/Integration logic
def load_data():
    if os.path.exists("warehouse_metrics.csv"):
        return pd.read_csv("warehouse_metrics.csv")
    return None

data = load_data()

if data is not None:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Steps", len(data))
    with col2:
        st.metric("Final Distance to Target", round(data['target_dist'].iloc[-1], 2))
    with col3:
        st.metric("Total Reward", round(data['reward'].sum(), 2))

    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Robot Trajectory (X, Y)")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='x', y='y', hue='step', palette='viridis', ax=ax)
        ax.set_title("AMR Path in Warehouse")
        st.pyplot(fig)

    with col_right:
        st.subheader("Convergence: Distance to Target")
        fig, ax = plt.subplots()
        sns.lineplot(data=data, x='step', y='target_dist', ax=ax, color='red')
        ax.set_title("Distance Optimization Over Time")
        st.pyplot(fig)

    st.subheader("Raw Simulation Logs")
    st.dataframe(data.tail(10))

else:
    st.warning("No simulation data found. Please run 'python train_amr.py' first.")

if st.button("Refresh Dashboard"):
    st.rerun()

st.sidebar.title("Configuration")
st.sidebar.info("This dashboard displays metrics from the 4-member Warehouse AMR Project.")
st.sidebar.button("Simulate Again (Trigger Process)")
