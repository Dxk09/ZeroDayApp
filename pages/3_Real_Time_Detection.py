import streamlit as st
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import json
from utils.visualization import VisualizationUtils
from utils.model_evaluation import ModelEvaluator
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Real-Time Detection - Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Real-Time Anomaly Detection")
st.markdown("Upload new data for real-time anomaly detection and monitoring.")

# Check if trained models are available
if not st.session_state.models:
    st.warning("No trained models available. Please train a model first.")
    st.info("👈 Go to **Model Training** page to train a model.")
    st.stop()

# Model selection
st.header("1. Model Selection")

col1, col2 = st.columns([2, 1])

with col1:
    selected_model_name = st.selectbox(
        "Select trained model for detection:",
        options=list(st.session_state.models.keys()),
        help="Choose a trained model for anomaly detection"
    )

with col2:
    if selected_model_name:
        model_info = st.session_state.models[selected_model_name]
        config = model_info['config']
        st.metric("Input Features", config['num_input_units'])
        if 'test_results' in model_info:
            st.metric("Model Accuracy", f"{model_info['test_results']['accuracy']:.3f}")

# Get selected model
selected_model = st.session_state.models[selected_model_name]['model']

# Real-time detection section
st.header("2. Upload Data for Detection")

# File upload for detection
uploaded_file = st.file_uploader(
    "Choose a file for anomaly detection",
    type=['csv', 'xlsx', 'xls', 'json'],
    help="Upload data in the same format as your training data"
)

# Manual data input option
with st.expander("✏️ Manual Data Input"):
    st.write("Enter feature values manually (comma-separated):")
    
    num_features = st.session_state.models[selected_model_name]['config']['num_input_units']
    manual_input = st.text_area(
        f"Enter {num_features} feature values:",
        placeholder="1.2, 0.5, -0.3, 2.1, 0.8, -1.0",
        help=f"Enter exactly {num_features} comma-separated numeric values"
    )
    
    if st.button("🔍 Detect from Manual Input"):
        try:
            values = [float(x.strip()) for x in manual_input.split(',')]
            if len(values) != num_features:
                st.error(f"Expected {num_features} values, got {len(values)}")
            else:
                # Reshape for single prediction
                X_manual = np.array(values).reshape(1, -1)
                
                # Make prediction
                prediction = selected_model.predict(X_manual)[0]
                probability = selected_model.get_anomaly_score(X_manual)[0]
                
                # Display result
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error(f"🚨 **ANOMALY DETECTED**")
                    else:
                        st.success(f"✅ **NORMAL**")
                
                with col2:
                    st.metric("Anomaly Score", f"{probability:.3f}")
                
                with col3:
                    confidence = max(probability, 1-probability)
                    st.metric("Confidence", f"{confidence:.3f}")
                
                # Add to detection results
                detection_result = {
                    'timestamp': datetime.now().isoformat(),
                    'prediction': int(prediction),
                    'anomaly_score': float(probability),
                    'input_values': values,
                    'model_used': selected_model_name,
                    'source': 'manual_input'
                }
                
                st.session_state.detection_results.append(detection_result)
                
                # Generate alert if anomaly
                if prediction == 1:
                    alert = {
                        'timestamp': datetime.now().timestamp(),
                        'type': 'Anomaly Detection',
                        'severity': 'High' if probability > 0.8 else 'Medium',
                        'message': f'Anomaly detected with score {probability:.3f}',
                        'model': selected_model_name,
                        'source': 'manual_input'
                    }
                    st.session_state.alerts.append(alert)
        
        except ValueError as e:
            st.error(f"Invalid input format: {str(e)}")

# File-based detection
if uploaded_file:
    try:
        # Load data
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_type == 'json':
            df = pd.read_json(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} samples for detection")
        
        # Display sample data
        with st.expander("📊 Sample Data", expanded=True):
            st.dataframe(df.head())
        
        # Check feature count
        expected_features = st.session_state.models[selected_model_name]['config']['num_input_units']
        
        if df.shape[1] != expected_features:
            st.error(f"Feature mismatch: Expected {expected_features} features, got {df.shape[1]}")
            st.info("Make sure your data has the same number of features as the training data")
        else:
            # Detection parameters
            col1, col2 = st.columns(2)
            
            with col1:
                anomaly_threshold = st.slider(
                    "Anomaly threshold",
                    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                    help="Probability threshold for classifying as anomaly"
                )
            
            with col2:
                batch_process = st.checkbox(
                    "Batch processing",
                    value=True,
                    help="Process all samples at once vs. one by one"
                )
            
            # Run detection
            if st.button("🚀 Run Detection", type="primary"):
                with st.spinner("Running anomaly detection..."):
                    
                    # Prepare data (assuming it's already preprocessed)
                    X_detect = df.values
                    
                    if batch_process:
                        # Batch prediction
                        predictions = selected_model.predict(X_detect)
                        probabilities = selected_model.get_anomaly_score(X_detect)
                        
                        # Apply custom threshold
                        custom_predictions = (probabilities > anomaly_threshold).astype(int)
                        
                    else:
                        # Simulate real-time processing
                        predictions = []
                        probabilities = []
                        custom_predictions = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, row in enumerate(df.values):
                            X_single = row.reshape(1, -1)
                            pred = selected_model.predict(X_single)[0]
                            prob = selected_model.get_anomaly_score(X_single)[0]
                            custom_pred = 1 if prob > anomaly_threshold else 0
                            
                            predictions.append(pred)
                            probabilities.append(prob)
                            custom_predictions.append(custom_pred)
                            
                            # Update progress
                            progress = (i + 1) / len(df)
                            progress_bar.progress(progress)
                            status_text.write(f"Processed {i+1}/{len(df)} samples")
                            
                            # Add small delay to simulate real-time
                            time.sleep(0.01)
                        
                        predictions = np.array(predictions)
                        probabilities = np.array(probabilities)
                        custom_predictions = np.array(custom_predictions)
                
                # Results summary
                st.header("3. Detection Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_samples = len(df)
                    st.metric("Total Samples", total_samples)
                
                with col2:
                    anomalies_detected = int(np.sum(custom_predictions))
                    st.metric("Anomalies Detected", anomalies_detected)
                
                with col3:
                    anomaly_rate = (anomalies_detected / total_samples) * 100
                    st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
                
                with col4:
                    avg_score = np.mean(probabilities)
                    st.metric("Avg Anomaly Score", f"{avg_score:.3f}")
                
                # Results visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Anomaly score distribution
                    viz = VisualizationUtils()
                    fig_scores = go.Figure()
                    
                    fig_scores.add_trace(go.Histogram(
                        x=probabilities,
                        nbinsx=50,
                        name='Anomaly Scores',
                        marker_color='lightblue'
                    ))
                    
                    fig_scores.add_vline(
                        x=anomaly_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Threshold"
                    )
                    
                    fig_scores.update_layout(
                        title="Anomaly Score Distribution",
                        xaxis_title="Anomaly Score",
                        yaxis_title="Frequency",
                        height=400
                    )
                    
                    st.plotly_chart(fig_scores, use_container_width=True)
                
                with col2:
                    # Detection timeline
                    if not batch_process:
                        timestamps = [datetime.now() - timedelta(seconds=(len(df)-i)*0.01) for i in range(len(df))]
                        
                        fig_timeline = go.Figure()
                        
                        fig_timeline.add_trace(go.Scatter(
                            x=list(range(len(probabilities))),
                            y=probabilities,
                            mode='lines+markers',
                            name='Anomaly Score',
                            marker=dict(
                                color=['red' if pred == 1 else 'green' for pred in custom_predictions],
                                size=6
                            )
                        ))
                        
                        fig_timeline.add_hline(
                            y=anomaly_threshold,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Threshold"
                        )
                        
                        fig_timeline.update_layout(
                            title="Real-Time Detection Timeline",
                            xaxis_title="Sample Index",
                            yaxis_title="Anomaly Score",
                            height=400
                        )
                        
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    else:
                        # Pie chart for batch results
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['Normal', 'Anomaly'],
                            values=[total_samples - anomalies_detected, anomalies_detected],
                            marker_colors=['green', 'red']
                        )])
                        
                        fig_pie.update_layout(
                            title="Detection Results Summary",
                            height=400
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                # Detailed results table
                with st.expander("📋 Detailed Results", expanded=False):
                    results_df = df.copy()
                    results_df['Anomaly_Score'] = probabilities
                    results_df['Prediction'] = custom_predictions
                    results_df['Classification'] = ['Anomaly' if p == 1 else 'Normal' for p in custom_predictions]
                    
                    # Filter options
                    col1, col2 = st.columns(2)
                    with col1:
                        show_filter = st.selectbox(
                            "Show:",
                            options=['All', 'Anomalies Only', 'Normal Only']
                        )
                    
                    with col2:
                        sort_by = st.selectbox(
                            "Sort by:",
                            options=['Index', 'Anomaly Score (High to Low)', 'Anomaly Score (Low to High)']
                        )
                    
                    # Apply filters
                    if show_filter == 'Anomalies Only':
                        filtered_df = results_df[results_df['Prediction'] == 1]
                    elif show_filter == 'Normal Only':
                        filtered_df = results_df[results_df['Prediction'] == 0]
                    else:
                        filtered_df = results_df
                    
                    # Apply sorting
                    if sort_by == 'Anomaly Score (High to Low)':
                        filtered_df = filtered_df.sort_values('Anomaly_Score', ascending=False)
                    elif sort_by == 'Anomaly Score (Low to High)':
                        filtered_df = filtered_df.sort_values('Anomaly_Score', ascending=True)
                    
                    st.dataframe(filtered_df)
                    
                    # Download results
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"anomaly_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Store results in session state
                for i, (pred, prob) in enumerate(zip(custom_predictions, probabilities)):
                    detection_result = {
                        'timestamp': datetime.now().isoformat(),
                        'prediction': int(pred),
                        'anomaly_score': float(prob),
                        'sample_index': i,
                        'model_used': selected_model_name,
                        'source': 'file_upload'
                    }
                    st.session_state.detection_results.append(detection_result)
                
                # Generate alerts for high-confidence anomalies
                high_confidence_anomalies = np.sum((custom_predictions == 1) & (probabilities > 0.8))
                if high_confidence_anomalies > 0:
                    alert = {
                        'timestamp': datetime.now().timestamp(),
                        'type': 'High-Confidence Anomalies',
                        'severity': 'High',
                        'message': f'{high_confidence_anomalies} high-confidence anomalies detected',
                        'model': selected_model_name,
                        'source': 'file_upload'
                    }
                    st.session_state.alerts.append(alert)
                
                st.success("✅ Detection completed and results stored!")
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Real-time monitoring section
st.header("4. Real-Time Monitoring")

col1, col2 = st.columns(2)

with col1:
    # Recent detections
    st.subheader("📊 Recent Detection Activity")
    
    if st.session_state.detection_results:
        recent_results = sorted(
            st.session_state.detection_results,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:10]
        
        for result in recent_results:
            timestamp = datetime.fromisoformat(result['timestamp'])
            status = "🚨 Anomaly" if result['prediction'] == 1 else "✅ Normal"
            score = result['anomaly_score']
            
            with st.container():
                st.write(f"{status} - Score: {score:.3f} - {timestamp.strftime('%H:%M:%S')}")
    else:
        st.info("No detection results yet.")

with col2:
    # Real-time alerts
    st.subheader("🚨 Recent Alerts")
    
    if st.session_state.alerts:
        recent_alerts = sorted(
            st.session_state.alerts,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:5]
        
        for alert in recent_alerts:
            timestamp = datetime.fromtimestamp(alert['timestamp'])
            severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            
            with st.container():
                st.write(f"{severity_color.get(alert['severity'], '⚪')} **{alert['type']}**")
                st.write(f"   {alert['message']} - {timestamp.strftime('%H:%M:%S')}")
    else:
        st.info("No alerts generated yet.")

# Live monitoring dashboard
if st.session_state.detection_results:
    st.subheader("📈 Live Monitoring Dashboard")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (5 seconds)", value=False)
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Real-time plots
    viz = VisualizationUtils()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Real-time detection plot
        fig_realtime = viz.plot_real_time_detection(st.session_state.detection_results)
        st.plotly_chart(fig_realtime, use_container_width=True)
    
    with col2:
        # Alert timeline
        fig_alerts = viz.plot_alert_timeline(st.session_state.alerts)
        st.plotly_chart(fig_alerts, use_container_width=True)

# Clear data options
st.header("5. Data Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Detection Results"):
        st.session_state.detection_results = []
        st.success("Detection results cleared!")
        st.rerun()

with col2:
    if st.button("🗑️ Clear Alerts"):
        st.session_state.alerts = []
        st.success("Alerts cleared!")
        st.rerun()

with col3:
    if st.button("🗑️ Clear All"):
        st.session_state.detection_results = []
        st.session_state.alerts = []
        st.success("All data cleared!")
        st.rerun()

# Instructions
with st.expander("📖 Instructions", expanded=False):
    st.markdown("""
    ### How to use real-time detection:
    
    1. **Select a trained model** from the dropdown
    2. **Upload new data** in CSV, Excel, or JSON format
    3. **Adjust the anomaly threshold** if needed (default: 0.5)
    4. **Run detection** to analyze the data
    5. **Monitor results** in real-time dashboard
    
    ### Data format requirements:
    - Must have the **same number of features** as training data
    - Features should be in the **same order** as training data
    - Data should be **preprocessed** in the same way as training data
    
    ### Understanding results:
    - **Anomaly Score**: Probability (0-1) that sample is anomalous
    - **Threshold**: Cutoff for classifying as anomaly
    - **Classification**: Final decision (Normal/Anomaly) based on threshold
    
    ### Monitoring features:
    - **Real-time visualization** of detection results
    - **Alert system** for high-confidence anomalies
    - **Historical tracking** of all detection sessions
    - **Auto-refresh** for continuous monitoring
    
    ### Manual input:
    - Enter feature values separated by commas
    - Useful for testing specific scenarios
    - Immediate feedback on single samples
    """)
