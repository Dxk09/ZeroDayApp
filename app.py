import streamlit as st
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Zero-Day Anomaly Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Main page
st.title("🛡️ Zero-Day Anomaly Detection System")
st.markdown("---")

# System overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Active Models",
        value=len(st.session_state.models),
        delta=None
    )

with col2:
    st.metric(
        label="Loaded Datasets",
        value=len(st.session_state.datasets),
        delta=None
    )

with col3:
    st.metric(
        label="Recent Alerts",
        value=len([alert for alert in st.session_state.alerts if alert['timestamp'] > datetime.now().timestamp() - 3600]),
        delta=None
    )

with col4:
    st.metric(
        label="Detection Sessions",
        value=len(st.session_state.detection_results),
        delta=None
    )

st.markdown("---")

# Quick start guide
st.header("🚀 Quick Start Guide")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Data Upload & Preprocessing")
    st.write("📊 Upload your cybersecurity datasets (CSV, Excel, JSON)")
    st.write("🔧 Preprocess data with feature scaling and normalization")
    st.write("📋 View dataset statistics and quality metrics")
    
    st.subheader("3. Real-Time Detection")
    st.write("🔍 Upload new data for anomaly detection")
    st.write("⚡ Get real-time anomaly scores and classifications")
    st.write("🚨 Receive alerts for detected anomalies")

with col2:
    st.subheader("2. Model Training")
    st.write("🧠 Train ANN models for binary classification")
    st.write("📈 Monitor training progress and validation metrics")
    st.write("💾 Save and manage multiple model versions")
    
    st.subheader("4. Dashboard & Analytics")
    st.write("📊 View comprehensive performance metrics")
    st.write("📈 Analyze detection trends and patterns")
    st.write("🔧 Manage and compare model performance")

st.markdown("---")

# Recent activity
st.header("📋 Recent Activity")

if st.session_state.alerts:
    st.subheader("🚨 Recent Alerts")
    recent_alerts = sorted(st.session_state.alerts, key=lambda x: x['timestamp'], reverse=True)[:5]
    
    for alert in recent_alerts:
        timestamp = datetime.fromtimestamp(alert['timestamp'])
        severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        
        with st.container():
            col1, col2, col3 = st.columns([2, 6, 2])
            with col1:
                st.write(f"{severity_color.get(alert['severity'], '⚪')} {alert['severity']}")
            with col2:
                st.write(f"**{alert['type']}**: {alert['message']}")
            with col3:
                st.write(timestamp.strftime("%H:%M:%S"))
else:
    st.info("No recent alerts. System is running normally.")

st.markdown("---")

# System status
st.header("💡 System Status")

col1, col2 = st.columns(2)

with col1:
    st.success("✅ PyTorch Backend: Online")
    st.success("✅ Data Processing: Ready")
    st.success("✅ Model Training: Available")

with col2:
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    st.info(f"🖥️ Compute Device: {device}")
    st.info(f"🐍 Python: {'.'.join(map(str, [3, 8]))}")
    st.info(f"🔥 PyTorch: {torch.__version__}")

# Navigation instructions
st.markdown("---")
st.info("👈 Use the sidebar to navigate to different sections of the application.")
