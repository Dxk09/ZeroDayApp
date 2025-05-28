import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from models.anomaly_detector import AnomalyDetectionModel
from models.ocsvm_detector import OCSDVMDetector
from utils.model_evaluation import ModelEvaluator
from utils.visualization import VisualizationUtils
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(
    page_title="Model Training - Anomaly Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Model Training")
st.markdown("Train deep neural network models for zero-day anomaly detection.")

# Check if processed datasets are available
processed_datasets = {
    name: data for name, data in st.session_state.datasets.items() 
    if isinstance(data, dict) and 'X' in data
}

if not processed_datasets:
    st.warning("No processed datasets available. Please upload and process a dataset first.")
    st.info("👈 Go to **Data Upload** page to upload and preprocess your data.")
    st.stop()

# Dataset selection
st.header("1. Dataset Selection")

col1, col2 = st.columns([2, 1])

with col1:
    selected_dataset = st.selectbox(
        "Select processed dataset for training:",
        options=list(processed_datasets.keys()),
        help="Choose a preprocessed dataset from the Data Upload page"
    )

with col2:
    if selected_dataset:
        data = processed_datasets[selected_dataset]
        st.metric("Features", data['X'].shape[1])
        st.metric("Samples", data['X'].shape[0])
        st.metric("Anomaly Ratio", f"{np.mean(data['y']):.1%}")

# Display dataset info
if selected_dataset:
    data = processed_datasets[selected_dataset]
    
    with st.expander("📊 Dataset Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Preprocessing Configuration:**")
            config = data['preprocessing_config']
            for key, value in config.items():
                st.write(f"- {key}: {value}")
        
        with col2:
            st.write("**Class Distribution:**")
            normal_count = int(np.sum(data['y'] == 0))
            anomaly_count = int(np.sum(data['y'] == 1))
            st.write(f"- Normal: {normal_count} ({normal_count/len(data['y']):.1%})")
            st.write(f"- Anomaly: {anomaly_count} ({anomaly_count/len(data['y']):.1%})")

# Model type selection
st.header("2. Model Type Selection")

model_type = st.selectbox(
    "Choose anomaly detection algorithm:",
    options=["Deep Neural Network (ANN)", "One-Class SVM (OCSVM)"],
    help="ANN: Deep learning approach, good for complex patterns. OCSVM: Classical ML, faster training, works well with normal data only."
)

# Model configuration
st.header("3. Model Configuration")

col1, col2 = st.columns(2)

with col1:
    if model_type == "Deep Neural Network (ANN)":
        st.subheader("🏗️ Architecture")
        
        # Get number of input features
        num_input_units = data['X'].shape[1]
        st.info(f"Input features: {num_input_units}")
        
        num_hidden_layers = st.slider(
            "Number of hidden layers",
            min_value=1, max_value=10, value=4,
            help="Number of hidden layers in the neural network"
        )
        
        num_units_per_layer = st.slider(
            "Units per hidden layer",
            min_value=4, max_value=512, value=64,
            help="Number of neurons in each hidden layer"
        )
    
    else:  # OCSVM
        st.subheader("🔧 OCSVM Configuration")
        
        # Get number of input features
        num_input_units = data['X'].shape[1]
        st.info(f"Input features: {num_input_units}")
        
        kernel = st.selectbox(
            "Kernel type",
            options=['rbf', 'linear', 'poly', 'sigmoid'],
            help="RBF: Good for non-linear patterns. Linear: Fast, works for linearly separable data."
        )
        
        gamma = st.selectbox(
            "Gamma parameter",
            options=['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            index=0,
            help="Controls influence of single training example. 'scale' is usually good."
        )
        
        nu = st.slider(
            "Nu parameter",
            min_value=0.01, max_value=0.5, value=0.05, step=0.01,
            help="Expected fraction of outliers in training data (0.05 = 5%)"
        )

with col2:
    st.subheader("⚙️ Training Parameters")
    
    if model_type == "Deep Neural Network (ANN)":
        num_epochs = st.slider(
            "Number of epochs",
            min_value=10, max_value=5000, value=1000,
            help="Number of training iterations"
        )
        
        batch_size = st.selectbox(
            "Batch size",
            options=[8, 16, 32, 64, 128],
            index=2,
            help="Number of samples per training batch"
        )
        
        learning_rate = st.selectbox(
            "Learning rate",
            options=[0.0001, 0.001, 0.01, 0.1],
            index=1,
            format_func=lambda x: f"{x:.4f}",
            help="Learning rate for the optimizer"
        )
        
        patience = st.slider(
            "Early stopping patience",
            min_value=10, max_value=200, value=50,
            help="Number of epochs to wait for improvement before stopping"
        )
    
    else:  # OCSVM
        st.info("OCSVM trains automatically - no additional parameters needed!")
        st.write("**Training approach:**")
        st.write("- Uses only normal data for training")
        st.write("- No epochs or batches required")
        st.write("- Fast convergence with optimization")
        
        # Add note about data usage
        st.warning("⚠️ OCSVM works best when trained only on normal (non-anomalous) data. The algorithm will learn what 'normal' looks like and detect deviations.")

# Data splitting configuration
st.header("4. Data Splitting")

col1, col2 = st.columns(2)

with col1:
    test_size = st.slider(
        "Test set size",
        min_value=0.1, max_value=0.5, value=0.2,
        help="Fraction of data to use for testing"
    )

with col2:
    val_size = st.slider(
        "Validation set size",
        min_value=0.1, max_value=0.5, value=0.2,
        help="Fraction of data to use for validation"
    )

# Model naming
model_name = st.text_input(
    "Model name",
    value=f"AnomalyDetector_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    help="Unique name for your trained model"
)

# Training section
st.header("5. Training")

if st.button("🚀 Start Training", type="primary"):
    if model_name in st.session_state.models:
        st.error("Model name already exists. Please choose a different name.")
    else:
        # Prepare data
        X = data['X'].values
        y = data['y'].values
        
        # Initialize model based on type
        if model_type == "Deep Neural Network (ANN)":
            # Split data for ANN (needs train/val/test)
            try:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                val_size_adjusted = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, 
                    random_state=42, stratify=y_temp
                )
                
                st.success(f"✅ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                
            except Exception as e:
                st.error(f"Error splitting data: {str(e)}")
                st.stop()
            
            # Initialize ANN model
            model = AnomalyDetectionModel(
                num_input_units=num_input_units,
                num_hidden_layers=num_hidden_layers,
                num_units_per_hidden_layer=num_units_per_layer
            )
        
        else:  # OCSVM
            # For OCSVM, use only normal data for training
            normal_indices = y == 0
            X_normal = X[normal_indices]
            
            # Split normal data for training and some for testing
            X_train_normal, X_test_normal = train_test_split(
                X_normal, test_size=0.2, random_state=42
            )
            
            # Create test set with both normal and anomaly data
            X_test = X[~normal_indices][:len(X_test_normal)]  # Same number of anomalies as normal test
            y_test_anomaly = y[~normal_indices][:len(X_test_normal)]
            
            # Combine for balanced test set
            X_test = np.vstack([X_test_normal, X_test])
            y_test = np.hstack([np.zeros(len(X_test_normal)), y_test_anomaly])
            
            st.success(f"✅ OCSVM Data prepared: Train={len(X_train_normal)} normal samples, Test={len(X_test)} samples")
            
            # Initialize OCSVM model
            model = OCSDVMDetector(
                kernel=kernel,
                gamma=gamma,
                nu=nu
            )
        
        # Training progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()
        chart_container = st.empty()
        
        # Training history for live plotting
        training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        def progress_callback(epoch, total_epochs, train_loss, val_loss, train_acc, val_acc):
            # Update progress
            progress = epoch / total_epochs
            progress_bar.progress(progress)
            
            # Update status
            status_text.write(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Update metrics
            with metrics_container.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train Loss", f"{train_loss:.4f}")
                with col2:
                    st.metric("Val Loss", f"{val_loss:.4f}")
                with col3:
                    st.metric("Train Acc", f"{train_acc:.3f}")
                with col4:
                    st.metric("Val Acc", f"{val_acc:.3f}")
            
            # Store history
            training_history['epochs'].append(epoch)
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_accuracy'].append(train_acc)
            training_history['val_accuracy'].append(val_acc)
            
            # Update live chart every 10 epochs
            if epoch % 10 == 0 or epoch == total_epochs:
                try:
                    with chart_container.container():
                        viz = VisualizationUtils()
                        fig = viz.plot_training_history(training_history, "Live Training Progress")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    # Continue training even if visualization fails
                    pass
        
        # Start training
        try:
            with st.spinner("Training model..."):
                history = model.train(
                    X_train, y_train, X_val, y_val,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    patience=patience,
                    progress_callback=progress_callback
                )
            
            st.success("🎉 Training completed successfully!")
            
            # Final evaluation
            st.header("5. Model Evaluation")
            
            # Evaluate on test set
            evaluator = ModelEvaluator()
            test_results = model.evaluate(X_test, y_test)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{test_results['accuracy']:.3f}")
            with col2:
                # Handle different key formats for classification report
                anomaly_key = '1' if '1' in test_results['classification_report'] else 1
                st.metric("Precision", f"{test_results['classification_report'][anomaly_key]['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{test_results['classification_report'][anomaly_key]['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{test_results['classification_report'][anomaly_key]['f1-score']:.3f}")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion matrix
                fig_cm = evaluator.plot_confusion_matrix(
                    y_test, test_results['predictions'],
                    "Test Set Confusion Matrix"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # ROC curve
                fig_roc = evaluator.plot_roc_curve(
                    y_test, test_results['probabilities'],
                    "Test Set ROC Curve"
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            
            # Precision-Recall curve
            fig_pr = evaluator.plot_precision_recall_curve(
                y_test, test_results['probabilities'],
                "Test Set Precision-Recall Curve"
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            
            # Save model
            model_path = f"models/{model_name}.pth"
            model.save_model(model_path)
            
            # Store model in session state
            st.session_state.models[model_name] = {
                'model': model,
                'model_path': model_path,
                'training_history': history,
                'test_results': test_results,
                'config': {
                    'num_input_units': num_input_units,
                    'num_hidden_layers': num_hidden_layers,
                    'num_units_per_layer': num_units_per_layer,
                    'dataset': selected_dataset,
                    'training_params': {
                        'num_epochs': num_epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'patience': patience
                    }
                },
                'created_at': datetime.now().isoformat()
            }
            
            # Store training history
            st.session_state.training_history[model_name] = history
            
            st.success(f"✅ Model saved as '{model_name}' and ready for use!")
            
            # Classification report
            with st.expander("📊 Detailed Classification Report"):
                report_df = pd.DataFrame(test_results['classification_report']).transpose()
                st.dataframe(report_df)
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            st.exception(e)

# Model comparison section
if len(st.session_state.models) > 1:
    st.header("6. Model Comparison & Accuracy Analysis")
    
    # User option to enable comparison
    show_comparison = st.checkbox(
        "🔍 Compare model accuracy and performance", 
        value=True,
        help="Enable detailed comparison between your trained models to see accuracy differences"
    )
    
    if show_comparison:
        # Select models to compare
        models_to_compare = st.multiselect(
            "Select models to compare:",
            options=list(st.session_state.models.keys()),
            default=list(st.session_state.models.keys())[:3],
            help="Choose which models you want to compare for accuracy analysis"
        )
        
        if len(models_to_compare) > 1:
            # Compare metrics
            evaluator = ModelEvaluator()
            comparison_data = {}
            model_types = {}
            
            for model_name in models_to_compare:
                model_info = st.session_state.models[model_name]
                if 'test_results' in model_info:
                    results = model_info['test_results']
                    
                    # Handle different key formats for classification report
                    anomaly_key = '1' if '1' in results['classification_report'] else 1
                    normal_key = '0' if '0' in results['classification_report'] else 0
                    
                    comparison_data[model_name] = {
                        'accuracy': results['accuracy'],
                        'precision': results['classification_report'][anomaly_key]['precision'],
                        'recall': results['classification_report'][anomaly_key]['recall'],
                        'f1_score': results['classification_report'][anomaly_key]['f1-score'],
                        'auc_score': results.get('auc_score', 0)
                    }
                    
                    # Determine model type
                    config = model_info.get('config', {})
                    if 'num_hidden_layers' in config:
                        model_types[model_name] = "ANN"
                    else:
                        model_types[model_name] = "OCSVM"
            
            if comparison_data:
                # Accuracy comparison highlight
                st.subheader("🎯 Accuracy Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Best performing model
                    best_model = max(comparison_data.keys(), key=lambda x: comparison_data[x]['accuracy'])
                    best_accuracy = comparison_data[best_model]['accuracy']
                    st.metric(
                        "🏆 Best Accuracy", 
                        f"{best_accuracy:.3f}",
                        f"{model_types.get(best_model, 'Unknown')} - {best_model}"
                    )
                
                with col2:
                    # Average accuracy
                    avg_accuracy = np.mean([data['accuracy'] for data in comparison_data.values()])
                    st.metric("📊 Average Accuracy", f"{avg_accuracy:.3f}")
                
                with col3:
                    # Accuracy range
                    min_acc = min([data['accuracy'] for data in comparison_data.values()])
                    max_acc = max([data['accuracy'] for data in comparison_data.values()])
                    accuracy_range = max_acc - min_acc
                    st.metric("📈 Accuracy Range", f"{accuracy_range:.3f}")
                
                # Model type performance breakdown
                st.subheader("🤖 Performance by Model Type")
                
                ann_models = [name for name, type_name in model_types.items() if type_name == "ANN"]
                ocsvm_models = [name for name, type_name in model_types.items() if type_name == "OCSVM"]
                
                if ann_models and ocsvm_models:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        ann_accuracies = [comparison_data[name]['accuracy'] for name in ann_models]
                        avg_ann_acc = np.mean(ann_accuracies)
                        st.success(f"🧠 **ANN Models**: {avg_ann_acc:.3f} avg accuracy ({len(ann_models)} models)")
                        for name in ann_models:
                            st.write(f"  • {name}: {comparison_data[name]['accuracy']:.3f}")
                    
                    with col2:
                        ocsvm_accuracies = [comparison_data[name]['accuracy'] for name in ocsvm_models]
                        avg_ocsvm_acc = np.mean(ocsvm_accuracies)
                        st.info(f"🔧 **OCSVM Models**: {avg_ocsvm_acc:.3f} avg accuracy ({len(ocsvm_models)} models)")
                        for name in ocsvm_models:
                            st.write(f"  • {name}: {comparison_data[name]['accuracy']:.3f}")
                    
                    # Performance comparison insight
                    if avg_ann_acc > avg_ocsvm_acc:
                        diff = avg_ann_acc - avg_ocsvm_acc
                        st.info(f"💡 **Insight**: ANN models perform {diff:.3f} better on average for your dataset")
                    elif avg_ocsvm_acc > avg_ann_acc:
                        diff = avg_ocsvm_acc - avg_ann_acc
                        st.info(f"💡 **Insight**: OCSVM models perform {diff:.3f} better on average for your dataset")
                    else:
                        st.info("💡 **Insight**: Both model types perform similarly on your dataset")
                
                # Radar chart comparison
                fig_comparison = evaluator.compare_models(comparison_data)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Detailed metrics table with highlighting
                st.subheader("📋 Detailed Performance Metrics")
                comparison_df = pd.DataFrame(comparison_data).T
                comparison_df['Model Type'] = [model_types.get(name, 'Unknown') for name in comparison_df.index]
                
                # Reorder columns
                columns = ['Model Type'] + [col for col in comparison_df.columns if col != 'Model Type']
                comparison_df = comparison_df[columns]
                
                # Style the dataframe to highlight best performers
                def highlight_best(s):
                    if s.name in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']:
                        is_max = s == s.max()
                        return ['background-color: #d4edda' if v else '' for v in is_max]
                    return ['' for _ in s]
                
                styled_df = comparison_df.round(3).style.apply(highlight_best)
                st.dataframe(styled_df, use_container_width=True)
                
                # Download comparison report
                if st.button("📥 Download Comparison Report"):
                    comparison_csv = comparison_df.round(3).to_csv()
                    st.download_button(
                        label="💾 Download CSV",
                        data=comparison_csv,
                        file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

# Available models sidebar
with st.sidebar:
    st.header("🧠 Trained Models")
    
    if st.session_state.models:
        for name, model_info in st.session_state.models.items():
            with st.expander(f"🤖 {name}"):
                config = model_info['config']
                st.write(f"**Architecture:** {config['num_hidden_layers']} layers")
                st.write(f"**Hidden units:** {config['num_units_per_layer']}")
                st.write(f"**Dataset:** {config['dataset']}")
                
                if 'test_results' in model_info:
                    results = model_info['test_results']
                    st.write(f"**Accuracy:** {results['accuracy']:.3f}")
                    st.write(f"**F1-Score:** {results['classification_report']['1']['f1-score']:.3f}")
                
                if st.button(f"🗑️ Delete", key=f"delete_{name}"):
                    # Remove model file
                    if os.path.exists(model_info['model_path']):
                        os.remove(model_info['model_path'])
                    
                    # Remove from session state
                    del st.session_state.models[name]
                    if name in st.session_state.training_history:
                        del st.session_state.training_history[name]
                    
                    st.rerun()
    else:
        st.info("No trained models yet.")

# Instructions
with st.expander("📖 Instructions", expanded=False):
    st.markdown("""
    ### How to train a model:
    
    1. **Select a processed dataset** from the dropdown
    2. **Configure the model architecture** (layers and units)
    3. **Set training parameters** (epochs, batch size, learning rate)
    4. **Configure data splitting** for train/validation/test sets
    5. **Give your model a unique name**
    6. **Start training** and monitor progress in real-time
    
    ### Model architecture tips:
    - **More layers**: Can capture complex patterns but may overfit
    - **More units**: Increases model capacity but requires more data
    - **Start simple**: Try 4 layers with 64 units each
    
    ### Training parameter tips:
    - **Learning rate**: 0.001 is usually a good starting point
    - **Batch size**: 32 works well for most datasets
    - **Epochs**: Start with 1000, use early stopping to prevent overfitting
    - **Patience**: 50 epochs is usually sufficient for early stopping
    
    ### Evaluation metrics:
    - **Accuracy**: Overall correctness (be careful with imbalanced data)
    - **Precision**: How many predicted anomalies are actually anomalies
    - **Recall**: How many actual anomalies were detected
    - **F1-Score**: Harmonic mean of precision and recall
    - **AUC**: Area under ROC curve (threshold-independent metric)
    """)
