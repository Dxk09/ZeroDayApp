import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.kdd_zero_day_filter import KDDZeroDayFilter
from models.ocsvm_detector import OCSVMDetector
from models.anomaly_detector import AnomalyDetectionModel

# Configure page
st.set_page_config(
    page_title="Zero-Day Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ocsvm_model' not in st.session_state:
    st.session_state.ocsvm_model = None
if 'ann_model' not in st.session_state:
    st.session_state.ann_model = None
if 'ann_training_data' not in st.session_state:
    st.session_state.ann_training_data = None
if 'ocsvm_training_data' not in st.session_state:
    st.session_state.ocsvm_training_data = None
if 'zero_day_data' not in st.session_state:
    st.session_state.zero_day_data = None
if 'ocsvm_training_complete' not in st.session_state:
    st.session_state.ocsvm_training_complete = False
if 'ann_training_complete' not in st.session_state:
    st.session_state.ann_training_complete = False
if 'ocsvm_results' not in st.session_state:
    st.session_state.ocsvm_results = None
if 'ann_results' not in st.session_state:
    st.session_state.ann_results = None

# Initialize zero-day filter
zero_day_filter = KDDZeroDayFilter()

# Title and description
st.title("🎯 Zero-Day Detection System")
st.markdown("**Compare ANN vs OCSVM approaches for zero-day attack detection**")
st.markdown("---")

# Main navigation
page = st.selectbox(
    "Choose Detection Method",
    ["🧠 ANN (Artificial Neural Network)", "🎯 OCSVM (One-Class SVM)"],
    help="Select between ANN and OCSVM approaches for zero-day detection"
)

if page == "🧠 ANN (Artificial Neural Network)":
    st.header("🧠 ANN Zero-Day Detection")
    st.markdown("**Train ANN on Known Attacks → Test against Zero-Day Attacks from KDDTest+**")
    
    # ANN workflow
    tab1, tab2, tab3 = st.tabs(["📚 Train on Known Attacks", "🎯 Test Zero-Day Detection", "📊 Results Analysis"])

    with tab1:
        st.subheader("📚 Step 1: Train ANN on Known Attacks")
        st.markdown("Train your neural network on known attack patterns from KDDTrain+")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Load KDDTrain+ Dataset", type="primary", key="ann_load_data"):
                with st.spinner("Loading KDDTrain+ dataset..."):
                    try:
                        # Load training data (reuse existing logic)
                        train_file = os.path.join('attached_assets', 'KDDTrain+ copy.txt')
                        if os.path.exists(train_file):
                            column_names = [
                                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                                'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                                'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
                            ]
                            
                            kdd_train = pd.read_csv(train_file, names=column_names, low_memory=False)
                            kdd_train = kdd_train.drop('difficulty', axis=1)  # Remove difficulty column
                            filtered_train = zero_day_filter.filter_training_data(kdd_train)
                            
                            # Preprocessing with one-hot encoding for categoricals
                            X_df = filtered_train.drop('attack_type', axis=1)
                            y_original = filtered_train['attack_type']
                            categorical_cols = list(X_df.select_dtypes(include=['object']).columns)
                            numeric_cols = [c for c in X_df.columns if c not in categorical_cols]
                            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                            X_cat = ohe.fit_transform(X_df[categorical_cols]) if categorical_cols else np.empty((len(X_df),0))
                            X_num = X_df[numeric_cols].values.astype(float) if numeric_cols else np.empty((len(X_df),0))
                            X_encoded = np.hstack([X_num, X_cat])
                            
                            # Create binary labels
                            y = (y_original != 'normal').astype(int)
                            
                            # Store training data
                            st.session_state.ann_training_data = {
                                'X': X_encoded,
                                'y': y.values if hasattr(y, 'values') else np.asarray(y),
                                'original_labels': y_original,
                                'onehot_encoder': ohe,
                                'categorical_cols': categorical_cols,
                                'numeric_cols': numeric_cols
                            }
                            
                            st.success("✅ KDDTrain+ dataset loaded successfully!")
                            st.write(f"**Dataset Info:** {X_encoded.shape[0]:,} samples, {X_encoded.shape[1]} features")
                            st.write(f"**Attack Rate:** {np.mean(y):.2%}")
                            
                            # Debug info
                            unique_attacks = y_original.unique()
                            st.write(f"**Attack Types:** {len(unique_attacks)} types")
                            st.write(f"**Normal Samples:** {np.sum(y == 0):,}")
                            st.write(f"**Attack Samples:** {np.sum(y == 1):,}")
                            
                        else:
                            st.error("KDDTrain+ file not found. Please check the file path.")
                            
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
        
        with col2:
            st.info("**ANN Training Focus:**\n- Learn patterns from known attacks\n- Normal + attack traffic\n- Supervised learning approach")
        
        # ANN training section
        if 'ann_training_data' in st.session_state and st.session_state.ann_training_data is not None:
            st.markdown("---")
            st.subheader("🧠 Configure and Train ANN")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Architecture:**")
                num_hidden_layers = st.slider("Hidden Layers", 3, 8, 6, key="ann_layers")
                num_units_per_layer = st.slider("Units per Layer", 64, 256, 128, key="ann_units")
                
                st.write("**Training Parameters:**")
                num_epochs = st.slider("Epochs", 100, 1000, 300, key="ann_epochs")
                batch_size = st.slider("Batch Size", 32, 512, 64, key="ann_batch")
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f", key="ann_lr")
            
            with col2:
                st.write("**Data Configuration:**")
                train_split = st.slider("Training Split", 0.6, 0.9, 0.8, key="ann_train_split")
                val_split = st.slider("Validation Split", 0.1, 0.3, 0.2, key="ann_val_split")
                
                st.write("**Training Info:**")
                st.write(f"Features: {st.session_state.ann_training_data['X'].shape[1]}")
                st.write(f"Samples: {st.session_state.ann_training_data['X'].shape[0]:,}")
                st.write(f"Attack Rate: {np.mean(st.session_state.ann_training_data['y']):.2%}")
            
            if st.button("🎯 Train ANN Model", type="primary", key="ann_train"):
                with st.spinner("Training ANN on known attacks..."):
                    try:
                        # Prepare data
                        X = st.session_state.ann_training_data['X']
                        y = st.session_state.ann_training_data['y']
                        
                        # Split data
                        from sklearn.model_selection import train_test_split
                        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-train_split, random_state=42)
                        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
                        
                        # Convert to numpy arrays (already arrays if one-hot encoded)
                        X_train_array = np.asarray(X_train)
                        y_train_array = np.asarray(y_train)
                        X_val_array = np.asarray(X_val)
                        y_val_array = np.asarray(y_val)
                        
                        # Create and train model
                        model = AnomalyDetectionModel(
                            num_input_units=X_train_array.shape[1],
                            num_hidden_layers=num_hidden_layers,
                            num_units_per_hidden_layer=num_units_per_layer
                        )
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        def progress_callback(epoch, total_epochs, train_loss, val_loss, train_acc, val_acc):
                            progress = epoch / total_epochs
                            progress_bar.progress(progress)
                            progress_text.text(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
                        
                        # Train model
                        history = model.train(
                            X_train_array, y_train_array,
                            X_val_array, y_val_array,
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            progress_callback=progress_callback
                        )
                        
                        # Store model and mark training complete
                        st.session_state.ann_model = model
                        st.session_state.ann_training_complete = True
                        
                        progress_bar.progress(1.0)
                        progress_text.text("Training completed!")
                        
                        st.success("✅ ANN training completed successfully!")
                        
                        # Show training results
                        st.subheader("📊 Training Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Final Training Accuracy", f"{history['train_accuracy'][-1]:.4f}")
                            st.metric("Final Validation Accuracy", f"{history['val_accuracy'][-1]:.4f}")
                        
                        with col2:
                            st.metric("Final Training Loss", f"{history['train_loss'][-1]:.4f}")
                            st.metric("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        st.error("Please check your parameters and try again.")

    with tab2:
        st.subheader("🎯 Step 2: Test ANN Against Zero-Day Attacks")
        st.markdown("Test your trained ANN against previously unseen zero-day attacks from KDDTest+")
        
        if not st.session_state.ann_training_complete:
            st.warning("⚠️ Please train your ANN model first in the 'Train on Known Attacks' tab")
            st.stop()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎯 Load Zero-Day Attacks and Test ANN", type="primary", key="ann_test"):
                with st.spinner("Testing ANN against zero-day attacks..."):
                    try:
                        # Load and test zero-day data (reuse existing logic)
                        test_file = os.path.join('attached_assets', 'KDDTest+ copy.txt')
                        if os.path.exists(test_file):
                            column_names = [
                                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                                'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                                'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
                            ]
                            
                            kdd_test = pd.read_csv(test_file, names=column_names, low_memory=False)
                            kdd_test = kdd_test.drop('difficulty', axis=1)
                            
                            # Filter to zero-day attacks
                            filtered_test = zero_day_filter.filter_zero_day_test_data(kdd_test, include_normal=True)
                            
                            # Preprocess data with one-hot encoding
                            X_test_df = filtered_test.drop('attack_type', axis=1)
                            y_test_original = filtered_test['attack_type']
                            ohe = st.session_state.ann_training_data['onehot_encoder']
                            cat_cols = st.session_state.ann_training_data['categorical_cols']
                            num_cols = st.session_state.ann_training_data['numeric_cols']
                            X_cat = ohe.transform(X_test_df[cat_cols]) if cat_cols else np.empty((len(X_test_df),0))
                            X_num = X_test_df[num_cols].values.astype(float) if num_cols else np.empty((len(X_test_df),0))
                            X_test = np.hstack([X_num, X_cat])
                            
                            # Create binary labels
                            y_test_binary = (y_test_original != 'normal').astype(int)
                            
                            # Make predictions using ANN
                            model = st.session_state.ann_model
                            X_test_array = X_test
                            predictions = model.predict(X_test_array)
                            anomaly_scores = model.get_anomaly_score(X_test_array)
                            
                            # Evaluate
                            evaluation_results = zero_day_filter.evaluate_zero_day_detection(
                                y_test_binary, predictions, y_test_original
                            )
                            
                            # Store results
                            st.session_state.ann_results = {
                                'predictions': predictions,
                                'anomaly_scores': anomaly_scores,
                                'true_labels': y_test_binary,
                                'original_labels': y_test_original,
                                'evaluation': evaluation_results,
                                'timestamp': datetime.now()
                            }
                            
                            st.success("✅ ANN zero-day detection test completed!")
                            
                            # Show results
                            overall = evaluation_results['overall']
                            zero_day_only_results = evaluation_results['zero_day_only']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Overall Accuracy", f"{overall['accuracy']:.3f}")
                            with col2:
                                st.metric("Zero-Day Detection Rate", f"{zero_day_only_results['zero_day_detection_rate']:.3f}")
                            with col3:
                                st.metric("Detected Zero-Day", f"{zero_day_only_results['detected_zero_day_attacks']}")
                            with col4:
                                st.metric("Missed Zero-Day", f"{zero_day_only_results['missed_zero_day_attacks']}")
                            
                        else:
                            st.error("KDDTest+ file not found.")
                            
                    except Exception as e:
                        st.error(f"Testing failed: {str(e)}")
        
        with col2:
            st.info("**Zero-Day Attacks Include:**\n- apache2, mailbomb, processtable\n- snmpgetattack, snmpguess\n- mscan, httptunnel, worm\n- sendmail, xlock, xsnoop\n- named, saint, udpstorm, ps")

    with tab3:
        st.subheader("📊 Step 3: ANN Results Analysis")
        st.markdown("Comprehensive analysis of ANN zero-day detection performance")
        
        if st.session_state.ann_results is None:
            st.warning("⚠️ Please run the zero-day detection test first")
        else:
            # Show detailed results
            results = st.session_state.ann_results
            evaluation = results['evaluation']
            
            st.subheader("🎯 Overall Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{evaluation['overall']['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{evaluation['overall']['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{evaluation['overall']['recall']:.3f}")

            # ROC, PR curves and exports for ANN
            y_true = results.get('true_labels')
            y_scores = results.get('anomaly_scores')
            if y_scores is not None and y_true is not None:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                auc_roc = auc(fpr, tpr)
                auc_pr = auc(recall, precision)

                st.markdown("---")
                st.subheader("📈 ROC & PR Curves")
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC AUC={auc_roc:.3f}'))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Chance', line=dict(dash='dash')))
                roc_fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', title='ROC Curve')
                st.plotly_chart(roc_fig, use_container_width=True)

                pr_fig = go.Figure()
                pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR AUC={auc_pr:.3f}'))
                pr_fig.update_layout(xaxis_title='Recall', yaxis_title='Precision', title='Precision-Recall Curve')
                st.plotly_chart(pr_fig, use_container_width=True)

                export_df = pd.DataFrame({
                    'anomaly_probability': y_scores,
                    'predicted_label': results['predictions'],
                    'true_label': results['true_labels'],
                    'original_attack_type': results['original_labels']
                })
                st.download_button('⬇️ Download ANN Predictions (CSV)', data=export_df.to_csv(index=False), file_name='ann_predictions.csv', mime='text/csv')

elif page == "🎯 OCSVM (One-Class SVM)":
    st.header("🎯 OCSVM Zero-Day Detection")
    st.markdown("**Train OCSVM on Normal Traffic → Test against Zero-Day Attacks from KDDTest+**")
    
    # OCSVM workflow
    tab1, tab2, tab3 = st.tabs(["📚 Train on Normal Traffic", "🎯 Test Zero-Day Detection", "📊 Results Analysis"])

    with tab1:
        st.subheader("📚 Step 1: Train OCSVM on Normal Traffic")
        st.markdown("Train your One-Class SVM on normal traffic patterns from KDDTrain+")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Load KDDTrain+ Normal Traffic", type="primary", key="ocsvm_load_data"):
                with st.spinner("Loading KDDTrain+ dataset..."):
                    try:
                        # Load KDDTrain+ data
                        train_file = os.path.join('attached_assets', 'KDDTrain+ copy.txt')
                        if os.path.exists(train_file):
                            # Load with proper column names
                            column_names = [
                                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                                'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                                'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
                            ]
                            
                            kdd_train = pd.read_csv(train_file, names=column_names, low_memory=False)
                            kdd_train = kdd_train.drop('difficulty', axis=1)
                            
                            # Filter to known attacks only
                            filtered_train = zero_day_filter.filter_training_data(kdd_train)
                            
                            # Preprocess data with one-hot encoding
                            X_df = filtered_train.drop('attack_type', axis=1)
                            y = filtered_train['attack_type']
                            categorical_cols = list(X_df.select_dtypes(include=['object']).columns)
                            numeric_cols = [c for c in X_df.columns if c not in categorical_cols]
                            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                            X_cat = ohe.fit_transform(X_df[categorical_cols]) if categorical_cols else np.empty((len(X_df),0))
                            X_num = X_df[numeric_cols].values.astype(float) if numeric_cols else np.empty((len(X_df),0))
                            X_encoded = np.hstack([X_num, X_cat])
                            
                            # Create binary labels (0=normal, 1=attack)
                            y_binary = (y != 'normal').astype(int)
                            
                            # Store training data
                            st.session_state.ocsvm_training_data = {
                                'X': X_encoded,
                                'y': y_binary.values if hasattr(y_binary, 'values') else np.asarray(y_binary),
                                'original_labels': y,
                                'onehot_encoder': ohe,
                                'categorical_cols': categorical_cols,
                                'numeric_cols': numeric_cols
                            }
                            
                            st.success("✅ KDDTrain+ dataset loaded successfully!")
                            st.write(f"**Dataset Info:** {X_encoded.shape[0]:,} samples, {X_encoded.shape[1]} features")
                            normal_samples = X_encoded.shape[0] - np.sum(y_binary)
                            st.write(f"**Normal Samples:** {int(normal_samples):,}")
                            st.write(f"**Normal Rate:** {(1 - np.mean(y_binary)):.2%}")
                            
                        else:
                            st.error("KDDTrain+ file not found. Please check the file path.")
                            
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
        
        with col2:
            st.info("**OCSVM Training Focus:**\n- Normal network traffic patterns\n- Learns baseline behavior\n- Detects deviations as anomalies")
        
        # Model training section
        if st.session_state.ocsvm_training_data is not None:
            st.markdown("---")
            st.subheader("🧠 Configure and Train OCSVM")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**OCSVM Parameters:**")
                kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], index=0, key="ocsvm_kernel")
                nu = st.slider("Nu (Outlier Fraction)", 0.01, 0.5, 0.1, step=0.01, key="ocsvm_nu")
                gamma = st.selectbox("Gamma", ['scale', 'auto', 0.001, 0.01, 0.1, 1.0], index=0, key="ocsvm_gamma")
                
            with col2:
                st.write("**Training Configuration:**")
                optimize_params = st.checkbox("Optimize Hyperparameters", value=False, key="ocsvm_optimize")
                
                st.write("**Training Info:**")
                st.write(f"Features: {st.session_state.ocsvm_training_data['X'].shape[1]}")
                st.write(f"Total Samples: {st.session_state.ocsvm_training_data['X'].shape[0]:,}")
                normal_samples = st.session_state.ocsvm_training_data['X'].shape[0] - np.sum(st.session_state.ocsvm_training_data['y'])
                st.write(f"Normal Samples: {normal_samples:,}")
                st.write(f"Normal Rate: {(1 - np.mean(st.session_state.ocsvm_training_data['y'])):.2%}")
            
            if st.button("🎯 Train OCSVM Model", type="primary", key="ocsvm_train"):
                with st.spinner("Training OCSVM on normal traffic..."):
                    try:
                        # Prepare data
                        X = st.session_state.ocsvm_training_data['X']
                        y = st.session_state.ocsvm_training_data['y']
                        
                        # Convert to numpy arrays (already arrays if one-hot encoded)
                        X_array = np.asarray(X)
                        y_array = np.asarray(y)
                        
                        # Create OCSVM model
                        model = OCSVMDetector(kernel=kernel, nu=nu, gamma=gamma)
                        
                        # Progress tracking
                        progress_text = st.empty()
                        
                        def progress_callback(message):
                            progress_text.text(message)
                        
                        # Train model (with or without hyperparameter optimization)
                        if optimize_params:
                            progress_callback("Optimizing hyperparameters...")
                            best_params, best_score = model.optimize_hyperparameters(X_array, y_array)
                            progress_callback(f"Best parameters found: {best_params}")
                            training_stats = model.training_stats
                        else:
                            training_stats = model.train(X_array, y_array, progress_callback=progress_callback)
                        
                        # Store model and mark training complete
                        st.session_state.ocsvm_model = model
                        st.session_state.ocsvm_training_complete = True
                        
                        progress_text.text("Training completed!")
                        
                        st.success("✅ OCSVM training completed successfully!")
                        
                        # Show training results
                        st.subheader("📊 Training Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Support Vectors", f"{training_stats['n_support_vectors']:,}")
                            st.metric("Support Vector Ratio", f"{training_stats['support_vector_ratio']:.4f}")
                        
                        with col2:
                            st.metric("Normal Samples Used", f"{training_stats['normal_samples_used']:,}")
                            st.metric("Total Samples Available", f"{training_stats['total_samples_available']:,}")
                        
                        # Show model parameters
                        st.subheader("🔧 Model Configuration")
                        param_col1, param_col2, param_col3 = st.columns(3)
                        
                        with param_col1:
                            st.write(f"**Kernel:** {model.kernel}")
                        with param_col2:
                            st.write(f"**Nu:** {model.nu}")
                        with param_col3:
                            st.write(f"**Gamma:** {model.gamma}")
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")

    with tab2:
        st.subheader("🎯 Step 2: Test OCSVM Against Zero-Day Attacks")
        st.markdown("Test your trained OCSVM against previously unseen zero-day attacks from KDDTest+")
        
        if not st.session_state.ocsvm_training_complete:
            st.warning("⚠️ Please train your OCSVM model first in the 'Train on Normal Traffic' tab")
            st.stop()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎯 Load Zero-Day Attacks and Test OCSVM", type="primary", key="ocsvm_test"):
                with st.spinner("Testing OCSVM against zero-day attacks..."):
                    try:
                        # Load zero-day test data
                        test_file = os.path.join('attached_assets', 'KDDTest+ copy.txt')
                        if os.path.exists(test_file):
                            column_names = [
                                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                                'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                                'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
                            ]
                            
                            kdd_test = pd.read_csv(test_file, names=column_names, low_memory=False)
                            kdd_test = kdd_test.drop('difficulty', axis=1)
                            
                            # Filter to zero-day attacks
                            filtered_test = zero_day_filter.filter_zero_day_test_data(kdd_test, include_normal=True)
                            
                            # Preprocess data with one-hot encoding
                            X_test_df = filtered_test.drop('attack_type', axis=1)
                            y_test_original = filtered_test['attack_type']
                            ohe = st.session_state.ocsvm_training_data['onehot_encoder']
                            cat_cols = st.session_state.ocsvm_training_data['categorical_cols']
                            num_cols = st.session_state.ocsvm_training_data['numeric_cols']
                            X_cat = ohe.transform(X_test_df[cat_cols]) if cat_cols else np.empty((len(X_test_df),0))
                            X_num = X_test_df[num_cols].values.astype(float) if num_cols else np.empty((len(X_test_df),0))
                            X_test = np.hstack([X_num, X_cat])
                            
                            # Create binary labels
                            y_test_binary = (y_test_original != 'normal').astype(int)
                            
                            # Make predictions using OCSVM
                            model = st.session_state.ocsvm_model
                            X_test_array = X_test
                            predictions, anomaly_scores = model.predict_binary(X_test_array, return_probabilities=True)
                            
                            # Evaluate using zero-day specific metrics
                            evaluation_results = zero_day_filter.evaluate_zero_day_detection(
                                y_test_binary, predictions, y_test_original
                            )
                            
                            # Store results
                            st.session_state.ocsvm_results = {
                                'predictions': predictions,
                                'anomaly_scores': anomaly_scores,
                                'true_labels': y_test_binary,
                                'original_labels': y_test_original,
                                'evaluation': evaluation_results,
                                'timestamp': datetime.now()
                            }
                            
                            st.success("✅ OCSVM zero-day detection test completed!")
                            
                            # Show key results
                            overall = evaluation_results['overall']
                            zero_day_only_results = evaluation_results['zero_day_only']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Overall Accuracy", f"{overall['accuracy']:.3f}")
                            with col2:
                                st.metric("Zero-Day Detection Rate", f"{zero_day_only_results['zero_day_detection_rate']:.3f}")
                            with col3:
                                st.metric("Detected Zero-Day", f"{zero_day_only_results['detected_zero_day_attacks']}")
                            with col4:
                                st.metric("Missed Zero-Day", f"{zero_day_only_results['missed_zero_day_attacks']}")
                            
                            # Performance summary
                            if zero_day_only_results['zero_day_detection_rate'] >= 0.9:
                                st.success("🎉 Excellent! Your OCSVM detected 90%+ of zero-day attacks")
                            elif zero_day_only_results['zero_day_detection_rate'] >= 0.8:
                                st.success("✅ Good! Your OCSVM detected 80%+ of zero-day attacks")
                            elif zero_day_only_results['zero_day_detection_rate'] >= 0.7:
                                st.warning("⚠️ Moderate. Your OCSVM detected 70%+ of zero-day attacks")
                            else:
                                st.error("❌ Poor. Your OCSVM detected less than 70% of zero-day attacks")
                            
                        else:
                            st.error("KDDTest+ file not found.")
                            
                    except Exception as e:
                        st.error(f"Testing failed: {str(e)}")
        
        with col2:
            st.info("**Zero-Day Attacks Include:**\n- apache2, mailbomb, processtable\n- snmpgetattack, snmpguess\n- mscan, httptunnel, worm\n- sendmail, xlock, xsnoop\n- named, saint, udpstorm, ps")

    with tab3:
        st.subheader("📊 Step 3: OCSVM Results Analysis")
        st.markdown("Comprehensive analysis of OCSVM zero-day detection performance")
        
        if st.session_state.ocsvm_results is None:
            st.warning("⚠️ Please run the zero-day detection test first")
        else:
            # Show detailed results
            results = st.session_state.ocsvm_results
            evaluation = results['evaluation']
            
            st.subheader("🎯 Overall Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{evaluation['overall']['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{evaluation['overall']['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{evaluation['overall']['recall']:.3f}")
            
            # Threshold tuning for OCSVM
            st.markdown("---")
            st.subheader("🔧 Threshold Tuning (OCSVM)")
            scores = results.get('anomaly_scores')
            if scores is not None:
                thr = st.slider("Anomaly threshold", 0.0, 1.0, 0.5, 0.01, key="ocsvm_threshold")
                y_true_thr = results['true_labels']
                y_pred_adj = (scores >= thr).astype(int)
                eval_adj = zero_day_filter.evaluate_zero_day_detection(y_true_thr, y_pred_adj, results['original_labels'])
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Adj Accuracy", f"{eval_adj['overall']['accuracy']:.3f}")
                with cols[1]:
                    st.metric("Adj Precision", f"{eval_adj['overall']['precision']:.3f}")
                with cols[2]:
                    st.metric("Adj Recall", f"{eval_adj['overall']['recall']:.3f}")
                with cols[3]:
                    st.metric("Adj F1", f"{eval_adj['overall']['f1_score']:.3f}")

            # ROC and PR curves + export for OCSVM
            y_true_plot = results['true_labels']
            y_scores_plot = results.get('anomaly_scores')
            if y_scores_plot is not None:
                fpr, tpr, _ = roc_curve(y_true_plot, y_scores_plot)
                precision, recall, _ = precision_recall_curve(y_true_plot, y_scores_plot)
                auc_roc = auc(fpr, tpr)
                auc_pr = auc(recall, precision)

                st.markdown("---")
                st.subheader("📈 ROC & PR Curves")
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC AUC={auc_roc:.3f}'))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Chance', line=dict(dash='dash')))
                roc_fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', title='ROC Curve')
                st.plotly_chart(roc_fig, use_container_width=True)

                pr_fig = go.Figure()
                pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR AUC={auc_pr:.3f}'))
                pr_fig.update_layout(xaxis_title='Recall', yaxis_title='Precision', title='Precision-Recall Curve')
                st.plotly_chart(pr_fig, use_container_width=True)

                export_df = pd.DataFrame({
                    'anomaly_probability': y_scores_plot,
                    'predicted_label': results['predictions'],
                    'true_label': results['true_labels'],
                    'original_attack_type': results['original_labels']
                })
                st.download_button('⬇️ Download OCSVM Predictions (CSV)', data=export_df.to_csv(index=False), file_name='ocsvm_predictions.csv', mime='text/csv')

            # Show attack-specific results
            st.subheader("📋 Attack-Specific Results")
            
            # evaluation uses 'by_attack_type' for per-attack metrics
            attack_results = evaluation.get('by_attack_type', {})
            attack_df = pd.DataFrame([
                {
                    'Attack Type': attack,
                    'Total Samples': stats['total_samples'],
                    'Detected': stats['detected_samples'],
                    'Detection Rate': f"{stats['detection_rate']:.3f}"
                }
                for attack, stats in attack_results.items()
            ])
            
            st.dataframe(attack_df, use_container_width=True)
