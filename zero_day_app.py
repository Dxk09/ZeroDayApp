import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.kdd_zero_day_filter import KDDZeroDayFilter
from models.ocsvm_detector import OCSVMDetector

# Configure page
st.set_page_config(
    page_title="Zero-Day ANN Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'zero_day_data' not in st.session_state:
    st.session_state.zero_day_data = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'zero_day_results' not in st.session_state:
    st.session_state.zero_day_results = None

# Initialize zero-day filter
zero_day_filter = KDDZeroDayFilter()

# Title and description
st.title("🎯 Zero-Day OCSVM Detection System")
st.markdown("**Train OCSVM on Normal Traffic → Test against Zero-Day Attacks from KDDTest+**")
st.markdown("---")

# Main workflow
tab1, tab2, tab3 = st.tabs(["📚 Train on Normal Traffic", "🎯 Test Zero-Day Detection", "📊 Results Analysis"])

with tab1:
    st.header("📚 Step 1: Train OCSVM on Normal Traffic")
    st.markdown("Train your One-Class SVM on normal traffic patterns from KDDTrain+")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Load KDDTrain+ Normal Traffic", type="primary"):
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
                        
                        # Preprocess data
                        X = filtered_train.drop('attack_type', axis=1)
                        y = filtered_train['attack_type']
                        
                        # Encode categorical features
                        label_encoders = {}
                        for col in X.select_dtypes(include=['object']).columns:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                            label_encoders[col] = le
                        
                        # Create binary labels (0=normal, 1=attack)
                        y_binary = (y != 'normal').astype(int)
                        
                        # Store training data
                        st.session_state.training_data = {
                            'X': X,
                            'y': y_binary,
                            'original_labels': y,
                            'label_encoders': label_encoders,
                            'analysis': {
                                'total_samples': len(filtered_train),
                                'normal_samples': len(filtered_train[filtered_train['attack_type'] == 'normal']),
                                'attack_samples': len(filtered_train[filtered_train['attack_type'] != 'normal']),
                                'known_attack_types': sorted(filtered_train[filtered_train['attack_type'] != 'normal']['attack_type'].unique())
                            }
                        }
                        
                        st.success(f"✅ KDDTrain+ loaded: {len(filtered_train)} samples")
                        
                        # Show training data analysis
                        analysis = st.session_state.training_data['analysis']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", f"{analysis['total_samples']:,}")
                        with col2:
                            st.metric("Normal Traffic", f"{analysis['normal_samples']:,}")
                        with col3:
                            st.metric("Known Attacks", f"{analysis['attack_samples']:,}")
                        
                        st.write("**Known Attack Types in Training:**")
                        st.write(", ".join(analysis['known_attack_types']))
                        
                    else:
                        st.error("KDDTrain+ file not found. Please check the file path.")
                        
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
    
    with col2:
        st.info("**OCSVM Training Focus:**\n- Normal network traffic patterns\n- Learns baseline behavior\n- Detects deviations as anomalies")
    
    # Model training section
    if st.session_state.training_data is not None:
        st.markdown("---")
        st.subheader("🧠 Configure and Train OCSVM")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**OCSVM Parameters:**")
            kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], index=0)
            nu = st.slider("Nu (Outlier Fraction)", 0.01, 0.5, 0.1, step=0.01)
            gamma = st.selectbox("Gamma", ['scale', 'auto', 0.001, 0.01, 0.1, 1.0], index=0)
            
        with col2:
            st.write("**Training Configuration:**")
            optimize_params = st.checkbox("Optimize Hyperparameters", value=False)
            
            st.write("**Training Info:**")
            st.write(f"Features: {len(st.session_state.training_data['X'].columns)}")
            st.write(f"Total Samples: {len(st.session_state.training_data['X']):,}")
            normal_samples = len(st.session_state.training_data['X']) - np.sum(st.session_state.training_data['y'])
            st.write(f"Normal Samples: {normal_samples:,}")
            st.write(f"Normal Rate: {(1 - np.mean(st.session_state.training_data['y'])):.2%}")
        
        if st.button("🎯 Train OCSVM Model", type="primary"):
            with st.spinner("Training OCSVM on normal traffic..."):
                try:
                    # Prepare data
                    X = st.session_state.training_data['X']
                    y = st.session_state.training_data['y']
                    
                    # Convert to numpy arrays
                    X_array = X.values
                    y_array = y.values
                    
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
                    st.session_state.model = model
                    st.session_state.training_complete = True
                    
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
    st.header("🎯 Step 2: Test Against Zero-Day Attacks")
    st.markdown("Test your trained OCSVM against previously unseen zero-day attacks from KDDTest+")
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train your OCSVM model first in the 'Train on Normal Traffic' tab")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎯 Load Zero-Day Attacks and Test", type="primary"):
            with st.spinner("Loading zero-day attacks from KDDTest+..."):
                try:
                    # Load KDDTest+ data
                    test_file = os.path.join('attached_assets', 'KDDTest+ copy.txt')
                    if os.path.exists(test_file):
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
                        
                        kdd_test = pd.read_csv(test_file, names=column_names, low_memory=False)
                        kdd_test = kdd_test.drop('difficulty', axis=1)
                        
                        # Filter to zero-day attacks only
                        filtered_test = zero_day_filter.filter_zero_day_test_data(kdd_test, include_normal=True)
                        
                        # Preprocess data using same encoders from training
                        X_test = filtered_test.drop('attack_type', axis=1)
                        y_test_original = filtered_test['attack_type']
                        
                        # Apply same encoding as training
                        label_encoders = st.session_state.training_data['label_encoders']
                        for col in X_test.select_dtypes(include=['object']).columns:
                            if col in label_encoders:
                                le = label_encoders[col]
                                # Handle unknown categories
                                X_test[col] = X_test[col].astype(str)
                                X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                                X_test[col] = le.transform(X_test[col])
                            else:
                                X_test[col] = 0  # Default encoding for unknown columns
                        
                        # Create binary labels (0=normal, 1=attack)
                        y_test_binary = (y_test_original != 'normal').astype(int)
                        
                        # Store zero-day data
                        st.session_state.zero_day_data = {
                            'X': X_test,
                            'y_binary': y_test_binary,
                            'original_labels': y_test_original,
                            'analysis': zero_day_filter.analyze_zero_day_distribution(filtered_test)
                        }
                        
                        st.success(f"✅ Zero-day test data loaded: {len(filtered_test)} samples")
                        
                        # Show zero-day data analysis
                        analysis = st.session_state.zero_day_data['analysis']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", f"{analysis['total_samples']:,}")
                        with col2:
                            st.metric("Normal Traffic", f"{analysis['attack_types'].get('normal', 0):,}")
                        with col3:
                            st.metric("Zero-Day Attacks", f"{analysis['zero_day_only']['total']:,}")
                        
                        st.write("**Zero-Day Attack Types:**")
                        zero_day_types = list(analysis['zero_day_only']['types'].keys())
                        st.write(", ".join(zero_day_types))
                        
                    else:
                        st.error("KDDTest+ file not found. Please check the file path.")
                        
                except Exception as e:
                    st.error(f"Error loading zero-day data: {str(e)}")
    
    with col2:
        st.info("**Zero-Day Attacks Include:**\n- apache2, mailbomb, processtable\n- snmpgetattack, snmpguess\n- mscan, httptunnel, worm\n- sendmail, xlock, xsnoop\n- named, saint, udpstorm, ps")
    
    # Testing section
    if st.session_state.zero_day_data is not None:
        st.markdown("---")
        st.subheader("🔍 Run Zero-Day Detection Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_normal = st.checkbox("Include Normal Traffic", value=True, help="Test false positive rate on normal traffic")
            zero_day_only = st.checkbox("Zero-Day Attacks Only", value=False, help="Focus only on zero-day attack detection")
        
        with col2:
            st.write("**Test Configuration:**")
            st.write(f"Test Samples: {len(st.session_state.zero_day_data['X']):,}")
            st.write(f"Zero-Day Types: {len(st.session_state.zero_day_data['analysis']['zero_day_only']['types'])}")
        
        if st.button("🚀 Run Zero-Day Detection Test", type="primary"):
            with st.spinner("Testing OCSVM against zero-day attacks..."):
                try:
                    X_test = st.session_state.zero_day_data['X']
                    y_test = st.session_state.zero_day_data['y_binary']
                    original_labels = st.session_state.zero_day_data['original_labels']
                    
                    # Filter data based on options
                    if zero_day_only:
                        zero_day_mask = original_labels != 'normal'
                        X_test = X_test[zero_day_mask]
                        y_test = y_test[zero_day_mask]
                        original_labels = original_labels[zero_day_mask]
                    
                    # Convert to numpy arrays for model prediction
                    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
                    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
                    
                    # Make predictions using OCSVM
                    model = st.session_state.model
                    predictions, anomaly_scores = model.predict_binary(X_test_array, return_probabilities=True)
                    
                    # Evaluate using zero-day specific metrics
                    evaluation_results = zero_day_filter.evaluate_zero_day_detection(
                        y_test_array, predictions, original_labels
                    )
                    
                    # Store results
                    st.session_state.zero_day_results = {
                        'predictions': predictions,
                        'anomaly_scores': anomaly_scores,
                        'true_labels': y_test_array,
                        'original_labels': original_labels,
                        'evaluation': evaluation_results,
                        'timestamp': datetime.now()
                    }
                    
                    st.success("✅ Zero-day detection test completed!")
                    
                    # Show key results
                    st.subheader("🎯 Zero-Day Detection Results")
                    
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
                    
                    # Quick performance summary
                    if zero_day_only_results['zero_day_detection_rate'] >= 0.9:
                        st.success("🎉 Excellent! Your OCSVM detected 90%+ of zero-day attacks")
                    elif zero_day_only_results['zero_day_detection_rate'] >= 0.8:
                        st.success("✅ Good! Your OCSVM detected 80%+ of zero-day attacks")
                    elif zero_day_only_results['zero_day_detection_rate'] >= 0.7:
                        st.warning("⚠️ Moderate. Your OCSVM detected 70%+ of zero-day attacks")
                    else:
                        st.error("❌ Poor. Your OCSVM detected less than 70% of zero-day attacks")
                        
                except Exception as e:
                    st.error(f"Testing failed: {str(e)}")

with tab3:
    st.header("📊 Step 3: Detailed Results Analysis")
    st.markdown("Comprehensive analysis of zero-day detection performance")
    
    if st.session_state.zero_day_results is None:
        st.warning("⚠️ Please run the zero-day detection test first")
        st.stop()
    
    results = st.session_state.zero_day_results
    evaluation = results['evaluation']
    
    # Overall performance metrics
    st.subheader("🎯 Overall Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Normal', 'Attack'],
                       y=['Normal', 'Attack'],
                       color_continuous_scale='Blues',
                       title="Confusion Matrix")
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                fig.add_annotation(x=j, y=i, text=str(cm[i, j]), 
                                 showarrow=False, font=dict(color="white" if cm[i, j] > cm.max()/2 else "black"))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Anomaly score distribution
        fig = px.histogram(
            x=results['anomaly_scores'],
            nbins=30,
            title="Anomaly Score Distribution",
            labels={'x': 'Anomaly Score', 'y': 'Count'}
        )
        fig.add_vline(x=0.5, line_dash="dash", annotation_text="Default Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # Zero-day specific analysis
    st.subheader("🎯 Zero-Day Attack Analysis")
    
    if 'by_attack_type' in evaluation:
        # Create DataFrame for attack-specific results
        attack_results = []
        for attack_type, metrics in evaluation['by_attack_type'].items():
            attack_results.append({
                'Attack Type': attack_type,
                'Total Samples': metrics['total_samples'],
                'Detected': metrics['detected_samples'],
                'Detection Rate': metrics['detection_rate'],
                'Zero-Day': 'Yes' if metrics['is_zero_day'] else 'No'
            })
        
        if attack_results:
            attack_df = pd.DataFrame(attack_results)
            
            # Sort by detection rate
            attack_df = attack_df.sort_values('Detection Rate', ascending=False)
            
            # Style the dataframe
            def highlight_zero_day(row):
                if row['Zero-Day'] == 'Yes':
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = attack_df.style.apply(highlight_zero_day, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Detection rate by attack type chart
            fig = px.bar(
                attack_df,
                x='Attack Type',
                y='Detection Rate',
                color='Zero-Day',
                title="Detection Rate by Attack Type",
                color_discrete_map={'Yes': '#dc3545', 'No': '#28a745'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics summary
    st.subheader("📊 Performance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Overall Metrics:**")
        overall = evaluation['overall']
        st.write(f"- Accuracy: {overall['accuracy']:.3f}")
        st.write(f"- Precision: {overall['precision']:.3f}")
        st.write(f"- Recall: {overall['recall']:.3f}")
        st.write(f"- F1-Score: {overall['f1_score']:.3f}")
        
        st.write(f"\n**Confusion Matrix:**")
        st.write(f"- True Positives: {overall['true_positives']}")
        st.write(f"- True Negatives: {overall['true_negatives']}")
        st.write(f"- False Positives: {overall['false_positives']}")
        st.write(f"- False Negatives: {overall['false_negatives']}")
    
    with col2:
        st.write("**Zero-Day Specific:**")
        zero_day_metrics = evaluation['zero_day_only']
        st.write(f"- Total Zero-Day Attacks: {zero_day_metrics['total_zero_day_attacks']}")
        st.write(f"- Detected Zero-Day: {zero_day_metrics['detected_zero_day_attacks']}")
        st.write(f"- Zero-Day Detection Rate: {zero_day_metrics['zero_day_detection_rate']:.3f}")
        st.write(f"- Missed Zero-Day: {zero_day_metrics['missed_zero_day_attacks']}")
        
        # Key insights
        st.write("\n**Key Insights:**")
        if zero_day_metrics['zero_day_detection_rate'] >= 0.9:
            st.success("🎉 Excellent zero-day detection capability")
        elif zero_day_metrics['zero_day_detection_rate'] >= 0.8:
            st.info("✅ Good zero-day detection performance")
        elif zero_day_metrics['zero_day_detection_rate'] >= 0.7:
            st.warning("⚠️ Moderate zero-day detection needs improvement")
        else:
            st.error("❌ Poor zero-day detection - model retraining recommended")
    
    # Export results
    st.subheader("💾 Export Results")
    
    if st.button("📥 Download Results Report"):
        # Create comprehensive report
        report_data = {
            'timestamp': results['timestamp'].isoformat(),
            'model_performance': evaluation['overall'],
            'zero_day_performance': evaluation['zero_day_only'],
            'by_attack_type': evaluation['by_attack_type']
        }
        
        import json
        report_json = json.dumps(report_data, indent=2)
        
        st.download_button(
            label="💾 Download JSON Report",
            data=report_json,
            file_name=f"zero_day_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Sidebar with model info
with st.sidebar:
    st.header("🎯 System Status")
    
    # Training status
    if st.session_state.training_complete:
        st.success("✅ ANN Model Trained")
        if st.session_state.training_data:
            analysis = st.session_state.training_data['analysis']
            st.write(f"Training samples: {analysis['total_samples']:,}")
            st.write(f"Known attacks: {len(analysis['known_attack_types'])}")
    else:
        st.error("❌ ANN Not Trained")
    
    # Zero-day data status
    if st.session_state.zero_day_data:
        st.success("✅ Zero-Day Data Loaded")
        analysis = st.session_state.zero_day_data['analysis']
        st.write(f"Test samples: {analysis['total_samples']:,}")
        st.write(f"Zero-day attacks: {analysis['zero_day_only']['total']:,}")
    else:
        st.error("❌ Zero-Day Data Not Loaded")
    
    # Results status
    if st.session_state.zero_day_results:
        st.success("✅ Testing Complete")
        zero_day_rate = st.session_state.zero_day_results['evaluation']['zero_day_only']['zero_day_detection_rate']
        st.write(f"Zero-day detection: {zero_day_rate:.1%}")
    else:
        st.error("❌ Testing Not Complete")
    
    st.markdown("---")
    st.markdown("**Workflow:**")
    st.markdown("1. Train ANN on KDDTrain+")
    st.markdown("2. Test against zero-day attacks")
    st.markdown("3. Analyze detection performance")
    
    if st.button("🔄 Reset All"):
        for key in ['model', 'training_data', 'zero_day_data', 'training_complete', 'zero_day_results']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("🎯 **Zero-Day ANN Detection** - Train on known attacks, test against unknown threats")