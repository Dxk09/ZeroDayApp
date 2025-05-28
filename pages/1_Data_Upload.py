import streamlit as st
import pandas as pd
import numpy as np
from utils.data_preprocessing import CyberSecurityDataProcessor
from utils.visualization import VisualizationUtils
import plotly.express as px

st.set_page_config(
    page_title="Data Upload - Anomaly Detection",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Upload & Preprocessing")
st.markdown("Upload and preprocess your cybersecurity datasets for anomaly detection training.")

# Initialize data processor
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = CyberSecurityDataProcessor()

# File upload section
st.header("1. Upload Dataset")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload CSV, Excel, or JSON files containing cybersecurity data"
    )

with col2:
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        st.info(f"File type: {file_type.upper()}")
        st.info(f"File size: {uploaded_file.size / 1024:.1f} KB")

# Sheet selection for Excel files
sheet_name = None
if uploaded_file and file_type in ['xlsx', 'xls']:
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        if len(sheet_names) > 1:
            sheet_name = st.selectbox("Select sheet:", sheet_names)
        else:
            sheet_name = sheet_names[0]
            st.info(f"Using sheet: {sheet_name}")
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")

# Load and analyze data
if uploaded_file:
    with st.spinner("Loading and analyzing dataset..."):
        # Load data
        df = st.session_state.data_processor.load_data(
            uploaded_file, file_type, sheet_name
        )
        
        if df is not None:
            # Store in session state
            dataset_name = uploaded_file.name
            st.session_state.datasets[dataset_name] = df
            
            # Analyze dataset
            analysis = st.session_state.data_processor.analyze_dataset(df)
            
            st.success(f"✅ Dataset loaded successfully: {analysis['shape'][0]} rows, {analysis['shape'][1]} columns")
            
            # Dataset overview
            st.header("2. Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", analysis['shape'][0])
            with col2:
                st.metric("Columns", analysis['shape'][1])
            with col3:
                st.metric("Numeric Features", len(analysis['numeric_columns']))
            with col4:
                st.metric("Categorical Features", len(analysis['categorical_columns']))
            
            # Display sample data
            with st.expander("📋 Sample Data (First 10 rows)", expanded=True):
                st.dataframe(df.head(10))
            
            # Column information
            with st.expander("📝 Column Information"):
                col_info = pd.DataFrame({
                    'Column': analysis['columns'],
                    'Data Type': [analysis['dtypes'][col] for col in analysis['columns']],
                    'Missing Values': [analysis['missing_values'][col] for col in analysis['columns']],
                    'Missing %': [f"{analysis['missing_percentage'][col]:.1f}%" for col in analysis['columns']],
                    'Unique Values': [analysis['unique_values'][col] for col in analysis['columns']]
                })
                st.dataframe(col_info)
            
            # Missing values visualization
            if any(analysis['missing_values'].values()):
                st.header("3. Data Quality Analysis")
                
                # Missing values heatmap
                missing_data = df.isnull()
                if missing_data.any().any():
                    fig_missing = px.imshow(
                        missing_data.T.astype(int),
                        title="Missing Values Heatmap",
                        color_continuous_scale='Reds',
                        aspect='auto'
                    )
                    fig_missing.update_layout(height=400)
                    st.plotly_chart(fig_missing, use_container_width=True)
                
                # Data quality report
                with st.expander("📊 Detailed Data Quality Report"):
                    quality_report = st.session_state.data_processor.generate_data_quality_report(df)
                    
                    st.subheader("Completeness Summary")
                    completeness_df = pd.DataFrame(quality_report['completeness']).T
                    st.dataframe(completeness_df)
                    
                    if quality_report['validity']:
                        st.subheader("Outlier Detection (Numeric Columns)")
                        validity_df = pd.DataFrame(quality_report['validity']).T
                        st.dataframe(validity_df)
            
            # Target column selection
            st.header("4. Target Column Selection")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if analysis['potential_targets']:
                    st.info(f"🎯 Suggested target columns: {', '.join(analysis['potential_targets'])}")
                
                target_column = st.selectbox(
                    "Select target column for anomaly detection:",
                    options=['None'] + analysis['columns'],
                    help="Choose the column that indicates normal vs anomalous behavior"
                )
                
                if target_column != 'None':
                    # Show target distribution
                    target_counts = df[target_column].value_counts()
                    st.write(f"**Target distribution:**")
                    for value, count in target_counts.items():
                        percentage = (count / len(df)) * 100
                        st.write(f"- {value}: {count} ({percentage:.1f}%)")
            
            with col2:
                if target_column != 'None':
                    # Target distribution pie chart
                    fig_target = px.pie(
                        values=df[target_column].value_counts().values,
                        names=df[target_column].value_counts().index,
                        title="Target Distribution"
                    )
                    st.plotly_chart(fig_target, use_container_width=True)
            
            # Data preprocessing options
            st.header("5. Preprocessing Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Missing Value Handling")
                missing_strategy = st.selectbox(
                    "Strategy for numeric columns:",
                    ['mean', 'median', 'mode'],
                    help="How to handle missing values in numeric columns"
                )
                
                st.subheader("Feature Scaling")
                scale_features = st.checkbox("Apply feature scaling", value=True)
                if scale_features:
                    scaling_method = st.selectbox(
                        "Scaling method:",
                        ['standard', 'minmax'],
                        help="Standard: zero mean, unit variance; MinMax: 0-1 range"
                    )
                else:
                    scaling_method = 'standard'
            
            with col2:
                st.subheader("Categorical Encoding")
                encode_categorical = st.checkbox("Encode categorical variables", value=True)
                
                st.subheader("Anomaly Label Configuration")
                if target_column != 'None':
                    unique_values = df[target_column].unique()
                    anomaly_values = st.multiselect(
                        "Select values that represent anomalies:",
                        options=unique_values,
                        default=[val for val in unique_values if str(val).lower() in ['1', 'attack', 'anomaly', 'malware', 'intrusion']][:1],
                        help="Choose which values in the target column represent anomalous behavior"
                    )
            
            # Process data
            if st.button("🔧 Process Dataset", type="primary"):
                if target_column == 'None':
                    st.error("Please select a target column before processing.")
                else:
                    with st.spinner("Processing dataset..."):
                        try:
                            # Preprocess features
                            X, y = st.session_state.data_processor.preprocess_features(
                                df,
                                target_column=target_column,
                                handle_missing=missing_strategy,
                                encode_categorical=encode_categorical,
                                scale_features=scale_features,
                                scaling_method=scaling_method
                            )
                            
                            # Prepare anomaly labels
                            y_binary = st.session_state.data_processor.prepare_anomaly_labels(
                                y, anomaly_indicators=anomaly_values
                            )
                            
                            # Store processed data
                            processed_data = {
                                'X': X,
                                'y': y_binary,
                                'feature_names': list(X.columns),
                                'original_target': y,
                                'preprocessing_config': {
                                    'target_column': target_column,
                                    'missing_strategy': missing_strategy,
                                    'encode_categorical': encode_categorical,
                                    'scale_features': scale_features,
                                    'scaling_method': scaling_method,
                                    'anomaly_values': anomaly_values
                                }
                            }
                            
                            # Store in session state
                            processed_name = f"{dataset_name}_processed"
                            st.session_state.datasets[processed_name] = processed_data
                            
                            st.success("✅ Dataset processed successfully!")
                            
                            # Show processed data summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Features", X.shape[1])
                            with col2:
                                normal_count = int(np.sum(y_binary == 0))
                                st.metric("Normal Samples", normal_count)
                            with col3:
                                anomaly_count = int(np.sum(y_binary == 1))
                                st.metric("Anomaly Samples", anomaly_count)
                            
                            # Class balance warning
                            if anomaly_count / len(y_binary) < 0.01:
                                st.warning("⚠️ Very imbalanced dataset detected. Consider using techniques like SMOTE for better training.")
                            elif anomaly_count / len(y_binary) > 0.4:
                                st.warning("⚠️ High anomaly ratio detected. Please verify your label configuration.")
                            
                            # Feature correlation analysis
                            if len(X.columns) > 1:
                                st.subheader("Feature Analysis")
                                
                                # Feature importance
                                try:
                                    importance_data = st.session_state.data_processor.get_feature_importance_data(X, y_binary)
                                    
                                    # Plot feature importance
                                    viz = VisualizationUtils()
                                    fig_importance = viz.plot_feature_importance(
                                        importance_data['mutual_info'],
                                        "Feature Importance (Mutual Information)"
                                    )
                                    st.plotly_chart(fig_importance, use_container_width=True)
                                    
                                except Exception as e:
                                    st.warning(f"Could not calculate feature importance: {str(e)}")
                                
                                # Correlation heatmap
                                if len(X.columns) <= 20:  # Only for manageable number of features
                                    fig_corr = viz.plot_correlation_heatmap(X)
                                    st.plotly_chart(fig_corr, use_container_width=True)
                            
                            # Data visualization
                            st.subheader("Data Distribution")
                            viz = VisualizationUtils()
                            fig_dist = viz.plot_data_distribution(X)
                            st.plotly_chart(fig_dist, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error processing dataset: {str(e)}")
                            st.exception(e)

# Available datasets sidebar
with st.sidebar:
    st.header("📚 Available Datasets")
    
    if st.session_state.datasets:
        for name, data in st.session_state.datasets.items():
            with st.expander(f"📄 {name}"):
                if isinstance(data, pd.DataFrame):
                    st.write(f"**Type:** Raw Dataset")
                    st.write(f"**Shape:** {data.shape}")
                    st.write(f"**Columns:** {len(data.columns)}")
                else:
                    st.write(f"**Type:** Processed Dataset")
                    st.write(f"**Features:** {data['X'].shape[1]}")
                    st.write(f"**Samples:** {data['X'].shape[0]}")
                    st.write(f"**Anomaly Ratio:** {np.mean(data['y']):.1%}")
                
                if st.button(f"🗑️ Remove", key=f"remove_{name}"):
                    del st.session_state.datasets[name]
                    st.rerun()
    else:
        st.info("No datasets loaded yet.")

# Instructions
with st.expander("📖 Instructions", expanded=False):
    st.markdown("""
    ### How to use this page:
    
    1. **Upload your dataset** in CSV, Excel, or JSON format
    2. **Review the dataset overview** to understand your data structure
    3. **Select the target column** that indicates normal vs anomalous behavior
    4. **Configure preprocessing options** based on your data requirements
    5. **Process the dataset** to prepare it for model training
    
    ### Supported data formats:
    - **CSV files** with headers
    - **Excel files** (single or multiple sheets)
    - **JSON files** with tabular structure
    
    ### Target column requirements:
    - Should contain binary or categorical values
    - Examples: 'normal'/'attack', 0/1, 'benign'/'malware'
    - Will be converted to binary labels (0=normal, 1=anomaly)
    
    ### Preprocessing features:
    - **Missing value handling** for data completeness
    - **Feature scaling** for neural network training
    - **Categorical encoding** for non-numeric features
    - **Anomaly label preparation** for supervised learning
    """)
