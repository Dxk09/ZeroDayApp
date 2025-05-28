import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
from datetime import datetime
from models.anomaly_detector import AnomalyDetectionModel
from utils.model_evaluation import ModelEvaluator
from utils.visualization import VisualizationUtils
import plotly.graph_objects as go
import json

st.set_page_config(
    page_title="Model Management - Anomaly Detection",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Model Management")
st.markdown("Manage, compare, and optimize your anomaly detection models.")

# Initialize utilities
evaluator = ModelEvaluator()
viz = VisualizationUtils()

# Model overview section
st.header("📋 Model Overview")

if not st.session_state.models:
    st.info("No trained models available.")
    st.markdown("👈 Go to **Model Training** page to train your first model.")
else:
    # Model summary table
    model_data = []
    for model_name, model_info in st.session_state.models.items():
        config = model_info['config']
        
        row = {
            'Model Name': model_name,
            'Architecture': f"{config['num_hidden_layers']} layers × {config['num_units_per_layer']} units",
            'Input Features': config['num_input_units'],
            'Dataset': config['dataset'],
            'Created': datetime.fromisoformat(model_info['created_at']).strftime('%Y-%m-%d %H:%M'),
        }
        
        # Add performance metrics if available
        if 'test_results' in model_info:
            results = model_info['test_results']
            row.update({
                'Accuracy': f"{results['accuracy']:.3f}",
                'F1-Score': f"{results['classification_report']['1']['f1-score']:.3f}",
                'AUC': f"{results.get('auc_score', 0):.3f}"
            })
        else:
            row.update({
                'Accuracy': 'N/A',
                'F1-Score': 'N/A', 
                'AUC': 'N/A'
            })
        
        model_data.append(row)
    
    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True)

# Model selection for detailed view
if st.session_state.models:
    st.header("🔍 Model Details")
    
    selected_model = st.selectbox(
        "Select model for detailed analysis:",
        options=list(st.session_state.models.keys()),
        help="Choose a model to view detailed information and performance metrics"
    )
    
    if selected_model:
        model_info = st.session_state.models[selected_model]
        config = model_info['config']
        
        # Model information tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🏗️ Architecture", 
            "📈 Performance", 
            "📋 Training History", 
            "⚙️ Management"
        ])
        
        with tab1:
            # Basic model information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Basic Information")
                st.write(f"**Model Name:** {selected_model}")
                st.write(f"**Created:** {datetime.fromisoformat(model_info['created_at']).strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Dataset Used:** {config['dataset']}")
                st.write(f"**Model File:** {model_info.get('model_path', 'Not saved')}")
                
                # Training parameters
                if 'training_params' in config:
                    st.subheader("🎯 Training Parameters")
                    params = config['training_params']
                    st.write(f"**Epochs:** {params.get('num_epochs', 'N/A')}")
                    st.write(f"**Batch Size:** {params.get('batch_size', 'N/A')}")
                    st.write(f"**Learning Rate:** {params.get('learning_rate', 'N/A')}")
                    st.write(f"**Patience:** {params.get('patience', 'N/A')}")
            
            with col2:
                st.subheader("🎯 Performance Summary")
                if 'test_results' in model_info:
                    results = model_info['test_results']
                    
                    # Performance metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Accuracy", f"{results['accuracy']:.3f}")
                        st.metric("Precision", f"{results['classification_report']['1']['precision']:.3f}")
                        st.metric("Recall", f"{results['classification_report']['1']['recall']:.3f}")
                    
                    with col_b:
                        st.metric("F1-Score", f"{results['classification_report']['1']['f1-score']:.3f}")
                        st.metric("AUC Score", f"{results.get('auc_score', 0):.3f}")
                        
                        # Model size estimation
                        total_params = sum(p.numel() for p in model_info['model'].model.parameters())
                        st.metric("Parameters", f"{total_params:,}")
                else:
                    st.info("No performance metrics available for this model.")
        
        with tab2:
            # Architecture details
            st.subheader("🏗️ Network Architecture")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Input Units:** {config['num_input_units']}")
                st.write(f"**Hidden Layers:** {config['num_hidden_layers']}")
                st.write(f"**Units per Layer:** {config['num_units_per_layer']}")
                st.write(f"**Output Units:** 1 (binary classification)")
                st.write(f"**Activation:** ReLU (hidden), Sigmoid (output)")
                st.write(f"**Dropout:** 0.2 (applied to hidden layers)")
            
            with col2:
                # Architecture visualization
                layers = []
                layer_sizes = []
                
                # Input layer
                layers.append("Input")
                layer_sizes.append(config['num_input_units'])
                
                # Hidden layers
                for i in range(config['num_hidden_layers']):
                    layers.append(f"Hidden {i+1}")
                    layer_sizes.append(config['num_units_per_layer'])
                
                # Output layer
                layers.append("Output")
                layer_sizes.append(1)
                
                # Create architecture diagram
                fig_arch = go.Figure()
                
                y_positions = list(range(len(layers)))
                x_positions = [size/10 for size in layer_sizes]  # Scale for visualization
                
                # Add nodes for each layer
                for i, (layer, size, x, y) in enumerate(zip(layers, layer_sizes, x_positions, y_positions)):
                    fig_arch.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode='markers+text',
                        marker=dict(size=max(20, min(100, size)), color='lightblue'),
                        text=f"{layer}<br>({size} units)",
                        textposition='middle center',
                        name=layer,
                        showlegend=False
                    ))
                
                # Add connections between layers
                for i in range(len(y_positions)-1):
                    fig_arch.add_trace(go.Scatter(
                        x=[x_positions[i], x_positions[i+1]],
                        y=[y_positions[i], y_positions[i+1]],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False
                    ))
                
                fig_arch.update_layout(
                    title="Network Architecture",
                    xaxis_title="Layer Width (scaled)",
                    yaxis_title="Layer Depth",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_arch, use_container_width=True)
        
        with tab3:
            # Performance analysis
            if 'test_results' in model_info:
                results = model_info['test_results']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confusion matrix
                    fig_cm = evaluator.plot_confusion_matrix(
                        results['confusion_matrix'].ravel(),  # Flatten if needed
                        results['predictions'],  # This might need adjustment based on stored format
                        title=f"Confusion Matrix - {selected_model}"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with col2:
                    # ROC curve
                    if 'probabilities' in results:
                        # Create a mock y_true for visualization (this should be stored properly)
                        # This is a limitation of the current storage format
                        st.info("ROC curve requires true labels to be stored with results")
                    else:
                        st.info("Probability scores not available for ROC curve")
                
                # Detailed classification report
                st.subheader("📊 Detailed Classification Report")
                if 'classification_report' in results:
                    report_df = pd.DataFrame(results['classification_report']).transpose()
                    st.dataframe(report_df.round(3), use_container_width=True)
                
                # Performance breakdown
                st.subheader("🎯 Performance Breakdown")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Normal Class (0):**")
                    normal_metrics = results['classification_report']['0']
                    st.write(f"- Precision: {normal_metrics['precision']:.3f}")
                    st.write(f"- Recall: {normal_metrics['recall']:.3f}")
                    st.write(f"- F1-Score: {normal_metrics['f1-score']:.3f}")
                    st.write(f"- Support: {normal_metrics['support']}")
                
                with col2:
                    st.write("**Anomaly Class (1):**")
                    anomaly_metrics = results['classification_report']['1']
                    st.write(f"- Precision: {anomaly_metrics['precision']:.3f}")
                    st.write(f"- Recall: {anomaly_metrics['recall']:.3f}")
                    st.write(f"- F1-Score: {anomaly_metrics['f1-score']:.3f}")
                    st.write(f"- Support: {anomaly_metrics['support']}")
                
                with col3:
                    st.write("**Overall:**")
                    macro_avg = results['classification_report']['macro avg']
                    st.write(f"- Macro Avg F1: {macro_avg['f1-score']:.3f}")
                    weighted_avg = results['classification_report']['weighted avg']
                    st.write(f"- Weighted Avg F1: {weighted_avg['f1-score']:.3f}")
                    st.write(f"- Accuracy: {results['accuracy']:.3f}")
            else:
                st.info("No performance metrics available for this model.")
        
        with tab4:
            # Training history
            if 'training_history' in model_info:
                history = model_info['training_history']
                
                # Plot training curves
                fig_history = viz.plot_training_history(history, f"Training History - {selected_model}")
                st.plotly_chart(fig_history, use_container_width=True)
                
                # Training statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Training Statistics")
                    st.write(f"**Epochs Completed:** {len(history['epochs'])}")
                    if history['train_loss']:
                        st.write(f"**Final Train Loss:** {history['train_loss'][-1]:.4f}")
                        st.write(f"**Best Train Loss:** {min(history['train_loss']):.4f}")
                    if history['val_loss'] and any(history['val_loss']):
                        st.write(f"**Final Val Loss:** {history['val_loss'][-1]:.4f}")
                        st.write(f"**Best Val Loss:** {min([l for l in history['val_loss'] if l > 0]):.4f}")
                
                with col2:
                    st.subheader("🎯 Accuracy Progress")
                    if history['train_accuracy']:
                        st.write(f"**Final Train Accuracy:** {history['train_accuracy'][-1]:.3f}")
                        st.write(f"**Best Train Accuracy:** {max(history['train_accuracy']):.3f}")
                    if history['val_accuracy'] and any(history['val_accuracy']):
                        st.write(f"**Final Val Accuracy:** {history['val_accuracy'][-1]:.3f}")
                        st.write(f"**Best Val Accuracy:** {max([a for a in history['val_accuracy'] if a > 0]):.3f}")
                
                # Training data export
                if st.button("📥 Export Training History"):
                    history_df = pd.DataFrame(history)
                    csv_data = history_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_data,
                        file_name=f"training_history_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No training history available for this model.")
        
        with tab5:
            # Model management operations
            st.subheader("⚙️ Model Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Export Model**")
                
                # Model export options
                export_format = st.selectbox(
                    "Export format:",
                    options=["PyTorch (.pth)", "Model Info (JSON)"],
                    help="Choose format for model export"
                )
                
                if st.button("📤 Export Model"):
                    if export_format == "PyTorch (.pth)":
                        # The model is already saved as .pth
                        model_path = model_info.get('model_path')
                        if model_path and os.path.exists(model_path):
                            with open(model_path, 'rb') as f:
                                st.download_button(
                                    label="💾 Download .pth file",
                                    data=f.read(),
                                    file_name=f"{selected_model}.pth",
                                    mime="application/octet-stream"
                                )
                        else:
                            st.error("Model file not found")
                    
                    elif export_format == "Model Info (JSON)":
                        # Export model configuration and metrics
                        export_data = {
                            'model_name': selected_model,
                            'config': config,
                            'created_at': model_info['created_at'],
                            'performance_metrics': model_info.get('test_results', {}),
                            'training_history': model_info.get('training_history', {})
                        }
                        
                        json_data = json.dumps(export_data, indent=2, default=str)
                        st.download_button(
                            label="💾 Download JSON",
                            data=json_data,
                            file_name=f"{selected_model}_info.json",
                            mime="application/json"
                        )
            
            with col2:
                st.write("**Model Actions**")
                
                # Model testing
                if st.button("🧪 Test Model", help="Test model with sample data"):
                    # This would require sample data - placeholder for now
                    st.info("Model testing requires sample data. Use the Real-Time Detection page for testing.")
                
                # Model deletion
                st.write("⚠️ **Danger Zone**")
                if st.button("🗑️ Delete Model", type="secondary"):
                    st.warning(f"Are you sure you want to delete model '{selected_model}'?")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✅ Confirm Delete"):
                            # Remove model file
                            model_path = model_info.get('model_path')
                            if model_path and os.path.exists(model_path):
                                os.remove(model_path)
                            
                            # Remove from session state
                            del st.session_state.models[selected_model]
                            if selected_model in st.session_state.training_history:
                                del st.session_state.training_history[selected_model]
                            
                            st.success(f"Model '{selected_model}' deleted successfully!")
                            st.rerun()
                    
                    with col_b:
                        if st.button("❌ Cancel"):
                            st.info("Deletion cancelled.")

# Model comparison section
if len(st.session_state.models) > 1:
    st.header("🔄 Model Comparison")
    
    # Select models to compare
    models_to_compare = st.multiselect(
        "Select models to compare:",
        options=list(st.session_state.models.keys()),
        default=list(st.session_state.models.keys())[:3],
        help="Choose 2 or more models for detailed comparison"
    )
    
    if len(models_to_compare) >= 2:
        # Comparison metrics
        comparison_data = {}
        architecture_data = []
        
        for model_name in models_to_compare:
            model_info = st.session_state.models[model_name]
            config = model_info['config']
            
            # Architecture comparison
            architecture_data.append({
                'Model': model_name,
                'Hidden Layers': config['num_hidden_layers'],
                'Units per Layer': config['num_units_per_layer'],
                'Total Parameters': sum(p.numel() for p in model_info['model'].model.parameters()),
                'Dataset': config['dataset']
            })
            
            # Performance comparison
            if 'test_results' in model_info:
                results = model_info['test_results']
                comparison_data[model_name] = {
                    'accuracy': results['accuracy'],
                    'precision': results['classification_report']['1']['precision'],
                    'recall': results['classification_report']['1']['recall'],
                    'f1_score': results['classification_report']['1']['f1-score'],
                    'auc_score': results.get('auc_score', 0)
                }
        
        # Architecture comparison table
        st.subheader("🏗️ Architecture Comparison")
        arch_df = pd.DataFrame(architecture_data)
        st.dataframe(arch_df, use_container_width=True)
        
        # Performance comparison
        if comparison_data:
            st.subheader("📊 Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Radar chart comparison
                fig_comparison = evaluator.compare_models(comparison_data)
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            with col2:
                # Metrics table
                metrics_df = pd.DataFrame(comparison_data).T
                metrics_df = metrics_df.round(3)
                
                # Highlight best performing models
                def highlight_max(s):
                    is_max = s == s.max()
                    return ['background-color: #d4edda' if v else '' for v in is_max]
                
                styled_df = metrics_df.style.apply(highlight_max, axis=0)
                st.dataframe(styled_df, use_container_width=True)
            
            # Best model recommendation
            if len(comparison_data) > 1:
                st.subheader("🏆 Recommendation")
                
                # Calculate overall score (weighted average of metrics)
                weights = {'accuracy': 0.3, 'precision': 0.2, 'recall': 0.2, 'f1_score': 0.25, 'auc_score': 0.05}
                overall_scores = {}
                
                for model_name, metrics in comparison_data.items():
                    score = sum(metrics[metric] * weight for metric, weight in weights.items() if metric in metrics)
                    overall_scores[model_name] = score
                
                best_model = max(overall_scores, key=overall_scores.get)
                best_score = overall_scores[best_model]
                
                st.success(f"🏆 **Recommended Model: {best_model}**")
                st.write(f"Overall Score: {best_score:.3f}")
                st.write("This model has the best weighted performance across all metrics.")

# Batch operations
if len(st.session_state.models) > 1:
    st.header("🔧 Batch Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Export All Models"):
            # Create zip file with all model information
            st.info("Batch export functionality would create a zip file with all models and their metadata.")
    
    with col2:
        if st.button("📊 Generate Comparison Report"):
            # Generate comprehensive comparison report
            if len(st.session_state.models) >= 2:
                st.info("Comprehensive comparison report generated (would be downloaded as PDF/HTML).")
            else:
                st.warning("Need at least 2 models for comparison report.")
    
    with col3:
        if st.button("🧹 Cleanup Unused Models"):
            # Identify and clean up unused models
            st.info("Model cleanup would identify models not used recently and offer to remove them.")

# Model storage and backup
st.header("💾 Storage Management")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Storage Usage")
    
    # Calculate storage usage
    total_models = len(st.session_state.models)
    total_size = 0
    
    for model_name, model_info in st.session_state.models.items():
        model_path = model_info.get('model_path')
        if model_path and os.path.exists(model_path):
            total_size += os.path.getsize(model_path)
    
    st.metric("Total Models", total_models)
    st.metric("Storage Used", f"{total_size / 1024 / 1024:.1f} MB")
    
    # List model files
    with st.expander("📁 Model Files"):
        for model_name, model_info in st.session_state.models.items():
            model_path = model_info.get('model_path')
            if model_path and os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / 1024 / 1024
                st.write(f"- {model_name}: {size_mb:.1f} MB")
            else:
                st.write(f"- {model_name}: File not found")

with col2:
    st.subheader("🔄 Backup Options")
    
    if st.button("💾 Backup All Models"):
        st.info("Backup functionality would create a compressed archive of all models and metadata.")
    
    if st.button("📥 Restore from Backup"):
        uploaded_backup = st.file_uploader(
            "Select backup file:",
            type=['zip', 'tar', 'gz'],
            help="Upload a previously created backup file"
        )
        
        if uploaded_backup:
            st.info("Backup restoration would extract and load models from the uploaded file.")

# Instructions
with st.expander("📖 Model Management Guide", expanded=False):
    st.markdown("""
    ### Model Management Features:
    
    **📋 Model Overview**
    - View all trained models in a comprehensive table
    - Compare key metrics and architecture details
    - Sort and filter models by various criteria
    
    **🔍 Detailed Model Analysis**
    - **Overview**: Basic information and performance summary
    - **Architecture**: Network structure and parameter details
    - **Performance**: Confusion matrix, ROC curves, and detailed metrics
    - **Training History**: Loss and accuracy curves over training epochs
    - **Management**: Export, test, and delete operations
    
    **🔄 Model Comparison**
    - Compare multiple models side-by-side
    - Radar charts for visual performance comparison
    - Architecture and parameter comparison tables
    - Automated recommendations for best performing models
    
    **⚙️ Management Operations**
    - **Export Models**: Save models in PyTorch or JSON format
    - **Delete Models**: Remove unwanted models and free storage
    - **Batch Operations**: Manage multiple models efficiently
    - **Storage Management**: Monitor disk usage and backup models
    
    ### Best Practices:
    - Regularly backup your trained models
    - Compare models before selecting for production use
    - Monitor storage usage and clean up unused models
    - Document model configurations and performance metrics
    - Use meaningful names for easy model identification
    
    ### Tips:
    - Models with higher F1-scores are generally better for imbalanced data
    - Consider both precision and recall when selecting models for production
    - Monitor training history to identify overfitting or underfitting
    - Keep multiple model versions for comparison and fallback
    """)
