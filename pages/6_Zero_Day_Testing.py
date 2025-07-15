import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.zero_day_simulator import ZeroDaySimulator
from utils.model_evaluation import ModelEvaluator
from utils.visualization import VisualizationUtils

st.set_page_config(page_title="Zero-Day Testing", page_icon="🎯", layout="wide")

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'zero_day_results' not in st.session_state:
    st.session_state.zero_day_results = {}
if 'attack_scenarios' not in st.session_state:
    st.session_state.attack_scenarios = {}

st.title("🎯 Zero-Day Attack Testing")
st.markdown("Test your models against realistic zero-day attacks and advanced evasion techniques")

# Initialize simulator
simulator = ZeroDaySimulator()
evaluator = ModelEvaluator()

# Sidebar for attack configuration
st.sidebar.header("Attack Configuration")
attack_descriptions = simulator.get_attack_descriptions()

# Check if models are available
available_models = list(st.session_state.models.keys())
if not available_models:
    st.error("No trained models available. Please train a model first in the Model Training page.")
    st.stop()

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Model for Testing",
    available_models,
    help="Choose which trained model to test against zero-day attacks"
)

# Attack type selection
attack_type = st.sidebar.selectbox(
    "Zero-Day Attack Type",
    list(attack_descriptions.keys()),
    format_func=lambda x: x.replace('_', ' ').title(),
    help="Select the type of zero-day attack to simulate"
)

# Attack configuration
st.sidebar.subheader("Attack Parameters")
num_samples = st.sidebar.slider(
    "Number of Attack Samples",
    min_value=10,
    max_value=500,
    value=100,
    help="Number of zero-day attack samples to generate"
)

include_evasion = st.sidebar.checkbox(
    "Include Evasion Techniques",
    value=True,
    help="Generate variants using advanced evasion techniques"
)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Attack Generation", "Detection Results", "Evasion Analysis", "Threat Intelligence"])

with tab1:
    st.header("🚀 Zero-Day Attack Generation")
    
    # Attack description
    st.info(f"**{attack_type.replace('_', ' ').title()}**: {attack_descriptions[attack_type]}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Zero-Day Attacks", type="primary"):
            with st.spinner("Generating sophisticated zero-day attacks..."):
                try:
                    # Generate zero-day attacks
                    zero_day_data, attack_labels = simulator.create_zero_day_test_suite(num_samples)
                    
                    # Store in session state
                    st.session_state.attack_scenarios[attack_type] = {
                        'data': zero_day_data,
                        'labels': attack_labels,
                        'timestamp': datetime.now(),
                        'parameters': {
                            'num_samples': num_samples,
                            'include_evasion': include_evasion
                        }
                    }
                    
                    st.success(f"Generated {len(zero_day_data)} zero-day attack samples")
                    
                    # Show preview
                    st.subheader("Attack Sample Preview")
                    preview_cols = ['duration', 'src_bytes', 'dst_bytes', 'protocol_type', 'service', 'flag']
                    st.dataframe(zero_day_data[preview_cols].head(10))
                    
                except Exception as e:
                    st.error(f"Error generating attacks: {str(e)}")
    
    with col2:
        if st.button("Test All Attack Types"):
            with st.spinner("Generating comprehensive zero-day test suite..."):
                try:
                    all_attacks = []
                    all_labels = []
                    
                    for attack_name in attack_descriptions.keys():
                        attack_data = simulator.generate_zero_day_attack(attack_name, num_samples // 4)
                        all_attacks.append(attack_data)
                        all_labels.extend([attack_name] * len(attack_data))
                    
                    combined_data = pd.concat(all_attacks, ignore_index=True)
                    
                    st.session_state.attack_scenarios['comprehensive'] = {
                        'data': combined_data,
                        'labels': all_labels,
                        'timestamp': datetime.now(),
                        'parameters': {
                            'num_samples': num_samples,
                            'attack_types': list(attack_descriptions.keys())
                        }
                    }
                    
                    st.success(f"Generated {len(combined_data)} samples across all attack types")
                    
                except Exception as e:
                    st.error(f"Error generating comprehensive suite: {str(e)}")

with tab2:
    st.header("🔍 Zero-Day Detection Results")
    
    if st.session_state.attack_scenarios:
        scenario = st.selectbox(
            "Select Attack Scenario",
            list(st.session_state.attack_scenarios.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if st.button("Run Detection Analysis", type="primary"):
            with st.spinner("Analyzing zero-day detection performance..."):
                try:
                    scenario_data = st.session_state.attack_scenarios[scenario]
                    attack_data = scenario_data['data']
                    
                    # Get the selected model
                    model = st.session_state.models[selected_model]
                    
                    # Make predictions
                    if hasattr(model, 'predict'):
                        predictions = model.predict(attack_data)
                        anomaly_scores = model.get_anomaly_score(attack_data)
                    else:
                        st.error("Selected model doesn't support prediction")
                        st.stop()
                    
                    # Calculate detection metrics
                    true_labels = np.ones(len(attack_data))  # All are anomalies
                    
                    # Store results
                    results = {
                        'predictions': predictions,
                        'anomaly_scores': anomaly_scores,
                        'true_labels': true_labels,
                        'attack_labels': scenario_data['labels'],
                        'timestamp': datetime.now()
                    }
                    
                    st.session_state.zero_day_results[f"{selected_model}_{scenario}"] = results
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    detection_rate = np.mean(predictions == 1)
                    false_negative_rate = np.mean(predictions == 0)
                    avg_anomaly_score = np.mean(anomaly_scores)
                    high_confidence_detections = np.sum(anomaly_scores > 0.7)
                    
                    with col1:
                        st.metric("Detection Rate", f"{detection_rate:.2%}")
                    with col2:
                        st.metric("False Negative Rate", f"{false_negative_rate:.2%}")
                    with col3:
                        st.metric("Avg Anomaly Score", f"{avg_anomaly_score:.3f}")
                    with col4:
                        st.metric("High Confidence", f"{high_confidence_detections}/{len(predictions)}")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Anomaly score distribution
                        fig = px.histogram(
                            x=anomaly_scores,
                            nbins=30,
                            title="Anomaly Score Distribution",
                            labels={'x': 'Anomaly Score', 'y': 'Frequency'}
                        )
                        fig.add_vline(x=0.5, line_dash="dash", annotation_text="Threshold")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Detection by attack type
                        if 'comprehensive' in scenario:
                            attack_type_df = pd.DataFrame({
                                'Attack Type': scenario_data['labels'],
                                'Detected': predictions,
                                'Score': anomaly_scores
                            })
                            
                            detection_by_type = attack_type_df.groupby('Attack Type')['Detected'].mean()
                            
                            fig = px.bar(
                                x=detection_by_type.index,
                                y=detection_by_type.values,
                                title="Detection Rate by Attack Type",
                                labels={'x': 'Attack Type', 'y': 'Detection Rate'}
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed analysis
                    st.subheader("Detailed Analysis")
                    
                    # Missed attacks (false negatives)
                    missed_attacks = attack_data[predictions == 0]
                    if len(missed_attacks) > 0:
                        st.warning(f"⚠️ {len(missed_attacks)} zero-day attacks were missed!")
                        
                        with st.expander("Analyze Missed Attacks"):
                            st.write("**Characteristics of missed attacks:**")
                            st.dataframe(missed_attacks.describe())
                            
                            # Show specific missed samples
                            st.write("**Sample missed attacks:**")
                            missed_indices = np.where(predictions == 0)[0]
                            for i, idx in enumerate(missed_indices[:5]):
                                st.write(f"**Attack {i+1}** (Score: {anomaly_scores[idx]:.3f})")
                                sample_data = attack_data.iloc[idx]
                                st.json(sample_data.to_dict())
                    
                    else:
                        st.success("✅ All zero-day attacks were successfully detected!")
                    
                except Exception as e:
                    st.error(f"Error in detection analysis: {str(e)}")
                    st.write("Debug info:", str(e))
    
    else:
        st.info("Generate attack scenarios first to see detection results")

with tab3:
    st.header("🎭 Evasion Analysis")
    
    if st.session_state.zero_day_results:
        result_key = st.selectbox(
            "Select Detection Result",
            list(st.session_state.zero_day_results.keys())
        )
        
        results = st.session_state.zero_day_results[result_key]
        
        st.subheader("Evasion Technique Effectiveness")
        
        # Analyze which evasion techniques were most effective
        missed_indices = np.where(results['predictions'] == 0)[0]
        detected_indices = np.where(results['predictions'] == 1)[0]
        
        if len(missed_indices) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Most Effective Evasion Patterns:**")
                # This would analyze the characteristics of missed attacks
                st.info("Attacks with low anomaly scores may indicate effective evasion")
                
                low_score_attacks = results['anomaly_scores'] < 0.3
                if np.any(low_score_attacks):
                    st.write(f"Found {np.sum(low_score_attacks)} attacks with very low anomaly scores")
            
            with col2:
                st.write("**Evasion Vulnerability Score:**")
                vulnerability_score = len(missed_indices) / len(results['predictions'])
                
                if vulnerability_score > 0.2:
                    st.error(f"High vulnerability: {vulnerability_score:.2%}")
                elif vulnerability_score > 0.1:
                    st.warning(f"Medium vulnerability: {vulnerability_score:.2%}")
                else:
                    st.success(f"Low vulnerability: {vulnerability_score:.2%}")
        
        # Robustness analysis
        st.subheader("Model Robustness Analysis")
        
        score_variance = np.var(results['anomaly_scores'])
        score_stability = 1 - (score_variance / np.mean(results['anomaly_scores']))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score Variance", f"{score_variance:.4f}")
        with col2:
            st.metric("Stability Index", f"{score_stability:.3f}")
        with col3:
            confidence_range = np.percentile(results['anomaly_scores'], 90) - np.percentile(results['anomaly_scores'], 10)
            st.metric("Confidence Range", f"{confidence_range:.3f}")

with tab4:
    st.header("🕵️ Threat Intelligence")
    
    st.subheader("Attack Pattern Analysis")
    
    if st.session_state.attack_scenarios:
        # Show attack characteristics
        for scenario_name, scenario_data in st.session_state.attack_scenarios.items():
            with st.expander(f"📊 {scenario_name.replace('_', ' ').title()} Analysis"):
                attack_data = scenario_data['data']
                
                # Statistical analysis
                st.write("**Attack Characteristics:**")
                stats_df = attack_data.describe()
                st.dataframe(stats_df)
                
                # Feature importance for this attack type
                if len(attack_data) > 10:
                    # Calculate feature variance as proxy for importance
                    feature_variance = attack_data.var().sort_values(ascending=False)
                    
                    fig = px.bar(
                        x=feature_variance.head(10).values,
                        y=feature_variance.head(10).index,
                        orientation='h',
                        title=f"Key Features for {scenario_name.replace('_', ' ').title()}",
                        labels={'x': 'Variance', 'y': 'Features'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Detection Recommendations")
    
    # Provide recommendations based on results
    if st.session_state.zero_day_results:
        st.write("**Model Improvement Recommendations:**")
        
        overall_detection_rate = 0
        total_attacks = 0
        
        for result_key, results in st.session_state.zero_day_results.items():
            detection_rate = np.mean(results['predictions'] == 1)
            overall_detection_rate += detection_rate * len(results['predictions'])
            total_attacks += len(results['predictions'])
        
        if total_attacks > 0:
            overall_detection_rate /= total_attacks
            
            if overall_detection_rate < 0.8:
                st.error("🚨 **Critical**: Detection rate below 80%. Consider:")
                st.write("- Retrain with more diverse attack patterns")
                st.write("- Adjust anomaly threshold")
                st.write("- Implement ensemble methods")
            
            elif overall_detection_rate < 0.9:
                st.warning("⚠️ **Moderate**: Detection rate needs improvement. Consider:")
                st.write("- Fine-tune model parameters")
                st.write("- Add more training data")
                st.write("- Implement feature engineering")
            
            else:
                st.success("✅ **Good**: Detection rate above 90%")
                st.write("- Continue monitoring for new attack patterns")
                st.write("- Regular model updates recommended")
    
    st.subheader("Zero-Day Threat Landscape")
    
    threat_info = {
        "Advanced Persistent Threat (APT)": "Nation-state level attacks with long-term persistence",
        "Polymorphic Malware": "Self-modifying code that evades signature-based detection",
        "AI-Powered Attacks": "Machine learning used to adapt attack patterns in real-time",
        "Supply Chain Attacks": "Compromise through trusted third-party components"
    }
    
    for threat, description in threat_info.items():
        st.write(f"**{threat}**: {description}")

# Footer
st.markdown("---")
st.markdown("🎯 **Zero-Day Testing Suite** - Validate your models against unknown threats")