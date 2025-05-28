import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.visualization import VisualizationUtils
from utils.model_evaluation import ModelEvaluator
import json

st.set_page_config(
    page_title="Dashboard - Anomaly Detection",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Anomaly Detection Dashboard")
st.markdown("Comprehensive analytics and monitoring for your cybersecurity anomaly detection system.")

# Initialize visualization utilities
viz = VisualizationUtils()
evaluator = ModelEvaluator()

# Check if there's any data to display
has_models = bool(st.session_state.models)
has_detection_results = bool(st.session_state.detection_results)
has_alerts = bool(st.session_state.alerts)

if not has_models and not has_detection_results and not has_alerts:
    st.info("No data available yet. Train some models and run detections to see analytics here.")
    st.markdown("### Getting Started:")
    st.markdown("1. 📊 Upload and preprocess your data")
    st.markdown("2. 🧠 Train anomaly detection models") 
    st.markdown("3. 🔍 Run real-time detection")
    st.markdown("4. 📈 Return here to view comprehensive analytics")
    st.stop()

# Time filter for dashboard
st.header("📅 Time Range Filter")

col1, col2, col3 = st.columns(3)

with col1:
    time_range = st.selectbox(
        "Select time range:",
        options=["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days", "All time"],
        index=2
    )

with col2:
    if time_range != "All time":
        # Calculate cutoff time
        if time_range == "Last 1 hour":
            cutoff = datetime.now() - timedelta(hours=1)
        elif time_range == "Last 6 hours":
            cutoff = datetime.now() - timedelta(hours=6)
        elif time_range == "Last 24 hours":
            cutoff = datetime.now() - timedelta(hours=24)
        elif time_range == "Last 7 days":
            cutoff = datetime.now() - timedelta(days=7)
        
        st.info(f"Showing data from {cutoff.strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    auto_refresh = st.checkbox("Auto-refresh (30 seconds)")
    if auto_refresh:
        st.empty()  # Placeholder for auto-refresh logic

# Filter data based on time range
def filter_by_time(data_list, time_field='timestamp'):
    if time_range == "All time":
        return data_list
    
    filtered_data = []
    for item in data_list:
        if time_field == 'timestamp' and isinstance(item.get(time_field), str):
            item_time = datetime.fromisoformat(item[time_field])
        elif time_field == 'timestamp' and isinstance(item.get(time_field), (int, float)):
            item_time = datetime.fromtimestamp(item[time_field])
        else:
            continue
            
        if item_time >= cutoff:
            filtered_data.append(item)
    
    return filtered_data

# Apply time filtering
if time_range != "All time":
    filtered_detection_results = filter_by_time(st.session_state.detection_results)
    filtered_alerts = filter_by_time(st.session_state.alerts)
else:
    filtered_detection_results = st.session_state.detection_results
    filtered_alerts = st.session_state.alerts

# System Overview Metrics
st.header("🎯 System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Active Models",
        value=len(st.session_state.models),
        help="Number of trained models available"
    )

with col2:
    total_detections = len(filtered_detection_results)
    st.metric(
        label="Total Detections",
        value=total_detections,
        help="Total number of detection runs in selected time range"
    )

with col3:
    if filtered_detection_results:
        anomalies_detected = sum(1 for r in filtered_detection_results if r.get('prediction') == 1)
        anomaly_rate = (anomalies_detected / total_detections) * 100 if total_detections > 0 else 0
        st.metric(
            label="Anomaly Rate",
            value=f"{anomaly_rate:.1f}%",
            help="Percentage of samples classified as anomalies"
        )
    else:
        st.metric("Anomaly Rate", "0.0%")

with col4:
    if filtered_alerts:
        critical_alerts = sum(1 for alert in filtered_alerts if alert.get('severity') == 'High')
        st.metric(
            label="Critical Alerts",
            value=critical_alerts,
            delta=f"{len(filtered_alerts)} total",
            help="High severity alerts in selected time range"
        )
    else:
        st.metric("Critical Alerts", "0")

st.markdown("---")

# Detection Analytics Section
if filtered_detection_results:
    st.header("🔍 Detection Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly score distribution
        scores = [r['anomaly_score'] for r in filtered_detection_results if 'anomaly_score' in r]
        predictions = [r['prediction'] for r in filtered_detection_results if 'prediction' in r]
        
        if scores and predictions:
            fig_dist = viz.plot_anomaly_distribution_by_time(filtered_detection_results, time_grouping='hour')
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Real-time detection timeline
        if len(filtered_detection_results) > 1:
            fig_timeline = viz.plot_real_time_detection(filtered_detection_results, time_window_minutes=1440)
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Need more detection data for timeline visualization")
    
    # Detailed score analysis
    if scores:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_score = np.mean(scores)
            st.metric("Average Score", f"{avg_score:.3f}")
        
        with col2:
            max_score = np.max(scores)
            st.metric("Highest Score", f"{max_score:.3f}")
        
        with col3:
            high_confidence = sum(1 for s in scores if s > 0.8)
            st.metric("High Confidence", f"{high_confidence}")

# Model Performance Section
if has_models:
    st.header("🧠 Model Performance")
    
    # Model comparison metrics
    model_metrics = {}
    for model_name, model_info in st.session_state.models.items():
        if 'test_results' in model_info:
            results = model_info['test_results']
            model_metrics[model_name] = {
                'accuracy': results['accuracy'],
                'precision': results['classification_report']['1']['precision'],
                'recall': results['classification_report']['1']['recall'],
                'f1_score': results['classification_report']['1']['f1-score'],
                'auc_score': results.get('auc_score', 0)
            }
    
    if model_metrics:
        # Model comparison radar chart
        if len(model_metrics) > 1:
            fig_comparison = evaluator.compare_models(model_metrics)
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Model performance table
        with st.expander("📋 Detailed Model Metrics", expanded=True):
            metrics_df = pd.DataFrame(model_metrics).T
            metrics_df = metrics_df.round(3)
            
            # Add color coding for performance
            def highlight_performance(val):
                if val >= 0.9:
                    return 'background-color: #d4edda'  # Light green
                elif val >= 0.8:
                    return 'background-color: #fff3cd'  # Light yellow
                elif val < 0.7:
                    return 'background-color: #f8d7da'  # Light red
                return ''
            
            styled_df = metrics_df.style.applymap(highlight_performance)
            st.dataframe(styled_df, use_container_width=True)
        
        # Model usage statistics
        if filtered_detection_results:
            model_usage = {}
            for result in filtered_detection_results:
                model_used = result.get('model_used', 'Unknown')
                model_usage[model_used] = model_usage.get(model_used, 0) + 1
            
            if model_usage:
                fig_usage = px.pie(
                    values=list(model_usage.values()),
                    names=list(model_usage.keys()),
                    title="Model Usage Distribution"
                )
                st.plotly_chart(fig_usage, use_container_width=True)

# Alert Analysis Section
if filtered_alerts:
    st.header("🚨 Alert Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Alert severity distribution
        severity_counts = {}
        for alert in filtered_alerts:
            severity = alert.get('severity', 'Unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        fig_severity = px.pie(
            values=list(severity_counts.values()),
            names=list(severity_counts.keys()),
            title="Alert Severity Distribution",
            color_discrete_map={
                'High': '#ff4444',
                'Medium': '#ffaa00', 
                'Low': '#44ff44'
            }
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        # Alert timeline
        fig_alert_timeline = viz.plot_alert_timeline(filtered_alerts, time_window_hours=24)
        st.plotly_chart(fig_alert_timeline, use_container_width=True)
    
    # Alert frequency analysis
    if len(filtered_alerts) > 0:
        # Group alerts by hour for frequency analysis
        alert_times = []
        for alert in filtered_alerts:
            if isinstance(alert.get('timestamp'), (int, float)):
                alert_times.append(datetime.fromtimestamp(alert['timestamp']))
            elif isinstance(alert.get('timestamp'), str):
                try:
                    alert_times.append(datetime.fromisoformat(alert['timestamp']))
                except:
                    continue
        
        if alert_times:
            # Create hourly frequency chart
            df_alerts = pd.DataFrame({'timestamp': alert_times})
            df_alerts['hour'] = df_alerts['timestamp'].dt.floor('H')
            hourly_counts = df_alerts.groupby('hour').size()
            
            fig_frequency = go.Figure(data=[
                go.Bar(x=hourly_counts.index, y=hourly_counts.values)
            ])
            fig_frequency.update_layout(
                title="Alert Frequency by Hour",
                xaxis_title="Time",
                yaxis_title="Number of Alerts",
                height=400
            )
            st.plotly_chart(fig_frequency, use_container_width=True)

# Data Quality and System Health
st.header("💊 System Health")

col1, col2, col3 = st.columns(3)

with col1:
    # Detection data quality
    if filtered_detection_results:
        complete_results = sum(1 for r in filtered_detection_results 
                             if all(key in r for key in ['prediction', 'anomaly_score', 'timestamp']))
        data_quality = (complete_results / len(filtered_detection_results)) * 100
        
        if data_quality >= 95:
            st.success(f"✅ Data Quality: {data_quality:.1f}%")
        elif data_quality >= 80:
            st.warning(f"⚠️ Data Quality: {data_quality:.1f}%")
        else:
            st.error(f"❌ Data Quality: {data_quality:.1f}%")
    else:
        st.info("No detection data to assess")

with col2:
    # Model health check
    if has_models:
        healthy_models = 0
        for model_name, model_info in st.session_state.models.items():
            if 'test_results' in model_info:
                accuracy = model_info['test_results']['accuracy']
                if accuracy >= 0.8:
                    healthy_models += 1
        
        model_health = (healthy_models / len(st.session_state.models)) * 100
        
        if model_health >= 80:
            st.success(f"✅ Model Health: {model_health:.0f}%")
        elif model_health >= 60:
            st.warning(f"⚠️ Model Health: {model_health:.0f}%")
        else:
            st.error(f"❌ Model Health: {model_health:.0f}%")
    else:
        st.info("No models to assess")

with col3:
    # System responsiveness
    if filtered_detection_results:
        recent_activity = sum(1 for r in filtered_detection_results 
                            if 'timestamp' in r)
        
        if recent_activity > 0:
            st.success(f"✅ System Active")
        else:
            st.warning("⚠️ No Recent Activity")
    else:
        st.info("No activity data")

# Performance Trends
if len(st.session_state.detection_results) > 10:
    st.header("📈 Performance Trends")
    
    # Calculate trends over time
    all_results = sorted(st.session_state.detection_results, 
                        key=lambda x: x.get('timestamp', ''))
    
    if len(all_results) >= 10:
        # Group by time periods
        batch_size = max(1, len(all_results) // 10)
        trends = []
        
        for i in range(0, len(all_results), batch_size):
            batch = all_results[i:i+batch_size]
            anomaly_rate = sum(1 for r in batch if r.get('prediction') == 1) / len(batch)
            avg_score = np.mean([r.get('anomaly_score', 0) for r in batch])
            
            trends.append({
                'batch': i // batch_size + 1,
                'anomaly_rate': anomaly_rate,
                'avg_score': avg_score
            })
        
        if trends:
            fig_trends = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Anomaly Rate Trend', 'Average Anomaly Score Trend'),
                vertical_spacing=0.1
            )
            
            batches = [t['batch'] for t in trends]
            
            fig_trends.add_trace(
                go.Scatter(x=batches, y=[t['anomaly_rate'] for t in trends], 
                          name='Anomaly Rate', line=dict(color='red')),
                row=1, col=1
            )
            
            fig_trends.add_trace(
                go.Scatter(x=batches, y=[t['avg_score'] for t in trends], 
                          name='Avg Score', line=dict(color='blue')),
                row=2, col=1
            )
            
            fig_trends.update_layout(height=500, title="Performance Trends Over Time")
            st.plotly_chart(fig_trends, use_container_width=True)

# Export and Reports
st.header("📋 Reports and Export")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Generate Summary Report"):
        # Create comprehensive summary
        report = {
            'generated_at': datetime.now().isoformat(),
            'time_range': time_range,
            'summary': {
                'total_models': len(st.session_state.models),
                'total_detections': len(filtered_detection_results),
                'total_alerts': len(filtered_alerts),
                'anomaly_rate': (sum(1 for r in filtered_detection_results if r.get('prediction') == 1) / len(filtered_detection_results) * 100) if filtered_detection_results else 0
            },
            'model_performance': model_metrics if 'model_metrics' in locals() else {},
            'alert_breakdown': severity_counts if 'severity_counts' in locals() else {}
        }
        
        # Display report
        st.json(report)
        
        # Download option
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="💾 Download Report (JSON)",
            data=report_json,
            file_name=f"anomaly_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

with col2:
    if filtered_detection_results and st.button("📥 Export Detection Data"):
        # Create CSV export
        export_data = []
        for result in filtered_detection_results:
            export_data.append({
                'timestamp': result.get('timestamp'),
                'prediction': result.get('prediction'),
                'anomaly_score': result.get('anomaly_score'),
                'model_used': result.get('model_used'),
                'source': result.get('source')
            })
        
        df_export = pd.DataFrame(export_data)
        csv_data = df_export.to_csv(index=False)
        
        st.download_button(
            label="💾 Download CSV",
            data=csv_data,
            file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col3:
    if filtered_alerts and st.button("🚨 Export Alert Data"):
        # Create alert export
        alert_data = []
        for alert in filtered_alerts:
            alert_data.append({
                'timestamp': alert.get('timestamp'),
                'type': alert.get('type'),
                'severity': alert.get('severity'),
                'message': alert.get('message'),
                'model': alert.get('model'),
                'source': alert.get('source')
            })
        
        df_alerts = pd.DataFrame(alert_data)
        csv_alerts = df_alerts.to_csv(index=False)
        
        st.download_button(
            label="💾 Download CSV",
            data=csv_alerts,
            file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Auto-refresh implementation
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()

# Instructions
with st.expander("📖 Dashboard Guide", expanded=False):
    st.markdown("""
    ### Dashboard Features:
    
    **📊 System Overview**
    - Monitor key metrics across your detection system
    - Track model performance and detection rates
    - View critical alert counts
    
    **🔍 Detection Analytics**
    - Analyze anomaly detection patterns over time
    - Review score distributions and confidence levels
    - Monitor detection frequency and trends
    
    **🧠 Model Performance**
    - Compare multiple model performance metrics
    - Track model usage and effectiveness
    - Identify best-performing models
    
    **🚨 Alert Analysis**
    - Review alert severity distributions
    - Monitor alert frequency patterns
    - Track alert response times
    
    **💊 System Health**
    - Monitor data quality and completeness
    - Check model health and accuracy
    - Verify system responsiveness
    
    **📈 Performance Trends**
    - Track performance changes over time
    - Identify patterns and anomalies in system behavior
    - Monitor long-term effectiveness
    
    ### Tips:
    - Use time range filters to focus on specific periods
    - Enable auto-refresh for real-time monitoring
    - Export data for external analysis and reporting
    - Monitor system health regularly for optimal performance
    """)
