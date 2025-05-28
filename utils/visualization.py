import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class VisualizationUtils:
    """Utilities for creating interactive visualizations"""
    
    @staticmethod
    def create_metric_cards(metrics):
        """Create metric cards data for Streamlit"""
        cards = []
        
        # Define metric configurations
        metric_configs = {
            'accuracy': {'label': 'Accuracy', 'format': '{:.3f}', 'color': 'blue'},
            'precision': {'label': 'Precision', 'format': '{:.3f}', 'color': 'green'},
            'recall': {'label': 'Recall', 'format': '{:.3f}', 'color': 'orange'},
            'f1_score': {'label': 'F1-Score', 'format': '{:.3f}', 'color': 'purple'},
            'auc_score': {'label': 'AUC Score', 'format': '{:.3f}', 'color': 'red'},
            'specificity': {'label': 'Specificity', 'format': '{:.3f}', 'color': 'teal'}
        }
        
        for metric_key, config in metric_configs.items():
            if metric_key in metrics:
                cards.append({
                    'label': config['label'],
                    'value': config['format'].format(metrics[metric_key]),
                    'delta': None,
                    'color': config['color']
                })
        
        return cards
    
    @staticmethod
    def plot_real_time_detection(detection_results, time_window_minutes=60):
        """Create real-time detection monitoring plot"""
        
        if not detection_results:
            # Empty state
            fig = go.Figure()
            fig.add_annotation(
                text="No detection data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Real-Time Anomaly Detection", height=400)
            return fig
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(detection_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter to time window
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        df_filtered = df[df['timestamp'] >= cutoff_time]
        
        # Create time series plot
        fig = go.Figure()
        
        # Plot anomaly scores
        fig.add_trace(go.Scatter(
            x=df_filtered['timestamp'],
            y=df_filtered['anomaly_score'],
            mode='lines+markers',
            name='Anomaly Score',
            line=dict(color='blue'),
            marker=dict(
                color=['red' if pred == 1 else 'green' for pred in df_filtered['prediction']],
                size=8
            )
        ))
        
        # Add threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Anomaly Threshold")
        
        fig.update_layout(
            title=f"Real-Time Anomaly Detection (Last {time_window_minutes} minutes)",
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_alert_timeline(alerts, time_window_hours=24):
        """Create alert timeline visualization"""
        
        if not alerts:
            fig = go.Figure()
            fig.add_annotation(
                text="No alerts in the selected time window",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Alert Timeline", height=300)
            return fig
        
        # Filter alerts by time window
        cutoff_time = datetime.now().timestamp() - (time_window_hours * 3600)
        filtered_alerts = [alert for alert in alerts if alert['timestamp'] >= cutoff_time]
        
        if not filtered_alerts:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No alerts in the last {time_window_hours} hours",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Alert Timeline", height=300)
            return fig
        
        # Convert to DataFrame
        df = pd.DataFrame(filtered_alerts)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Color mapping for severity
        color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}
        
        fig = go.Figure()
        
        for severity in df['severity'].unique():
            severity_data = df[df['severity'] == severity]
            fig.add_trace(go.Scatter(
                x=severity_data['datetime'],
                y=[severity] * len(severity_data),
                mode='markers',
                name=f'{severity} Severity',
                marker=dict(
                    color=color_map.get(severity, 'gray'),
                    size=12,
                    symbol='diamond'
                ),
                text=severity_data['message'],
                hovertemplate='<b>%{y} Severity</b><br>Time: %{x}<br>Message: %{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Alert Timeline (Last {time_window_hours} hours)",
            xaxis_title="Time",
            yaxis_title="Severity",
            height=300,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_data_distribution(df, columns=None, max_columns=6):
        """Create data distribution plots"""
        
        if columns is None:
            # Select numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_columns[:max_columns]
        
        if not columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No numeric columns available for distribution plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Data Distribution", height=400)
            return fig
        
        # Calculate subplot layout
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=columns,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, col in enumerate(columns):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            title="Feature Distributions",
            height=200 * n_rows + 100,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(df, columns=None):
        """Create correlation heatmap"""
        
        if columns is None:
            # Select numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_columns
        
        if len(columns) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 numeric columns for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Correlation Heatmap", height=400)
            return fig
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            height=600,
            width=600
        )
        
        return fig
    
    @staticmethod
    def plot_anomaly_distribution_by_time(detection_results, time_grouping='hour'):
        """Plot anomaly distribution over time"""
        
        if not detection_results:
            fig = go.Figure()
            fig.add_annotation(
                text="No detection data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Anomaly Distribution Over Time", height=400)
            return fig
        
        df = pd.DataFrame(detection_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by time
        if time_grouping == 'hour':
            df['time_group'] = df['timestamp'].dt.floor('H')
        elif time_grouping == 'day':
            df['time_group'] = df['timestamp'].dt.date
        else:  # minute
            df['time_group'] = df['timestamp'].dt.floor('T')
        
        # Count anomalies and normals by time group
        grouped = df.groupby(['time_group', 'prediction']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        if 0 in grouped.columns:
            fig.add_trace(go.Bar(
                x=grouped.index,
                y=grouped[0],
                name='Normal',
                marker_color='green'
            ))
        
        if 1 in grouped.columns:
            fig.add_trace(go.Bar(
                x=grouped.index,
                y=grouped[1],
                name='Anomaly',
                marker_color='red'
            ))
        
        fig.update_layout(
            title=f"Anomaly Distribution by {time_grouping.title()}",
            xaxis_title="Time",
            yaxis_title="Count",
            barmode='stack',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_system_status_indicators(models, datasets, recent_alerts):
        """Create system status indicators"""
        
        # Model status
        model_status = "🟢 Online" if models else "🔴 No Models Loaded"
        
        # Data status
        data_status = "🟢 Ready" if datasets else "🟡 No Data Loaded"
        
        # Alert status
        recent_high_alerts = len([a for a in recent_alerts if a.get('severity') == 'High'])
        if recent_high_alerts > 0:
            alert_status = f"🔴 {recent_high_alerts} High Severity"
        elif len(recent_alerts) > 0:
            alert_status = f"🟡 {len(recent_alerts)} Alerts"
        else:
            alert_status = "🟢 No Alerts"
        
        return {
            'models': model_status,
            'data': data_status,
            'alerts': alert_status
        }
