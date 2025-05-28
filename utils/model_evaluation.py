import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelEvaluator:
    """Comprehensive model evaluation utilities"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'specificity': self._calculate_specificity(y_true, y_pred),
        }
        
        if y_prob is not None:
            metrics['auc_score'] = roc_auc_score(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=['Normal', 'Anomaly'], output_dict=True
        )
        
        self.evaluation_results[model_name] = metrics
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (True Negative Rate)"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Create interactive confusion matrix plot"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Normal', 'Predicted Anomaly'],
            y=['Actual Normal', 'Actual Anomaly'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            height=400
        )
        
        return fig
    
    def plot_roc_curve(self, y_true, y_prob, title="ROC Curve"):
        """Create ROC curve plot"""
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_prob, title="Precision-Recall Curve"):
        """Create Precision-Recall curve plot"""
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400
        )
        
        return fig
    
    def plot_training_history(self, history, title="Training History"):
        """Plot training and validation metrics over epochs"""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Loss', 'Accuracy'),
            vertical_spacing=0.1
        )
        
        epochs = history['epochs']
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        if history['val_loss'] and any(history['val_loss']):
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_loss'], name='Validation Loss', line=dict(color='red')),
                row=1, col=1
            )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_accuracy'], name='Train Accuracy', line=dict(color='blue')),
            row=2, col=1
        )
        if history['val_accuracy'] and any(history['val_accuracy']):
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_accuracy'], name='Validation Accuracy', line=dict(color='red')),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_feature_importance(self, importance_dict, title="Feature Importance"):
        """Plot feature importance"""
        
        features = list(importance_dict.keys())
        values = list(importance_dict.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, values), key=lambda x: x[1], reverse=True)
        features_sorted, values_sorted = zip(*sorted_data)
        
        fig = go.Figure(go.Bar(
            x=values_sorted[:20],  # Top 20 features
            y=features_sorted[:20],
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600
        )
        
        return fig
    
    def plot_anomaly_scores_distribution(self, scores, labels, title="Anomaly Scores Distribution"):
        """Plot distribution of anomaly scores for normal vs anomalous samples"""
        
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=normal_scores,
            name='Normal',
            opacity=0.7,
            nbinsx=50,
            marker_color='green'
        ))
        
        fig.add_trace(go.Histogram(
            x=anomaly_scores,
            name='Anomaly',
            opacity=0.7,
            nbinsx=50,
            marker_color='red'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Anomaly Score",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def compare_models(self, model_results):
        """Compare multiple models side by side"""
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        model_names = list(model_results.keys())
        
        comparison_data = []
        for metric in metrics:
            metric_values = []
            for model_name in model_names:
                if metric in model_results[model_name]:
                    metric_values.append(model_results[model_name][metric])
                else:
                    metric_values.append(0)
            comparison_data.append(metric_values)
        
        fig = go.Figure()
        
        for i, model_name in enumerate(model_names):
            fig.add_trace(go.Scatterpolar(
                r=[comparison_data[j][i] for j in range(len(metrics))],
                theta=metrics,
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Model Comparison Radar Chart",
            height=500
        )
        
        return fig
    
    def generate_evaluation_report(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """Generate comprehensive evaluation report"""
        
        metrics = self.calculate_metrics(y_true, y_pred, y_prob, model_name)
        
        report = {
            'model_name': model_name,
            'metrics': metrics,
            'sample_size': len(y_true),
            'class_distribution': {
                'normal': int(np.sum(y_true == 0)),
                'anomaly': int(np.sum(y_true == 1))
            },
            'false_positives': int(np.sum((y_pred == 1) & (y_true == 0))),
            'false_negatives': int(np.sum((y_pred == 0) & (y_true == 1))),
            'true_positives': int(np.sum((y_pred == 1) & (y_true == 1))),
            'true_negatives': int(np.sum((y_pred == 0) & (y_true == 0)))
        }
        
        return report
