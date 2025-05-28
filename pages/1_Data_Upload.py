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

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = CyberSecurityDataProcessor()

if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

# Data source selection
st.header("1. Choose Data Source")

data_source = st.radio(
    "Select your data source:",
    ["📚 Use Built-in KDD Cup Dataset", "📤 Upload Custom File"],
    help="Choose between ready-to-use KDD Cup datasets or upload your own cybersecurity data"
)

if data_source == "📚 Use Built-in KDD Cup Dataset":
    st.subheader("🎯 KDD Cup 1999 - Network Intrusion Detection Dataset")
    
    # Dataset selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dataset_choice = st.selectbox(
            "Select KDD dataset:",
            ["KDD Cup 1999 (Training)", "KDD Cup 1999 (Testing)"],
            help="Training set for model development, Testing set for final evaluation"
        )
        
        # Show dataset information
        if dataset_choice == "KDD Cup 1999 (Training)":
            st.info("📊 **125,974 network connection records** with 41 features including protocol, service, duration, and network statistics")
            st.info("🎯 **23 different attack types** including DoS, R2L, U2R, and Probe categories")
            dataset_file = "attached_assets/KDDTrain+ copy.txt"
        else:
            st.info("📊 **22,545 test records** for model evaluation and validation")
            st.info("🎯 **Contains unknown attack types** perfect for zero-day detection testing")
            dataset_file = "attached_assets/KDDTest+ copy.txt"
    
    with col2:
        st.write("**Dataset Features:**")
        st.write("• Duration, Protocol, Service")
        st.write("• Network traffic statistics")
        st.write("• Host-based features") 
        st.write("• Content features")
        st.write("• Attack type labels")
    
    # Load KDD dataset
    if st.button("🚀 Load KDD Dataset", type="primary"):
        with st.spinner("Loading KDD dataset..."):
            try:
                # Define column names for KDD dataset
                kdd_columns = [
                    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
                ]
                
                # Load the dataset
                df = pd.read_csv(dataset_file, names=kdd_columns, header=None)
                
                # Remove difficulty column (not needed for detection)
                df = df.drop('difficulty', axis=1)
                
                # Store in session state
                dataset_name = f"KDD_{dataset_choice.split('(')[1].replace(')', '').strip()}"
                st.session_state.datasets[dataset_name] = df
                
                # Automatically process for training
                with st.spinner("Processing KDD dataset for training..."):
                    # Separate features and target
                    X = df.drop('attack_type', axis=1)
                    y = df['attack_type']
                    
                    # Convert categorical columns to numeric
                    from sklearn.preprocessing import LabelEncoder
                    label_encoders = {}
                    
                    for col in X.select_dtypes(include=['object']).columns:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        label_encoders[col] = le
                    
                    # Create binary labels (0=normal, 1=anomaly)
                    y_binary = (y != 'normal').astype(int)
                    
                    # Store processed data ready for training
                    processed_data = {
                        'X': X,
                        'y': y_binary,
                        'feature_names': list(X.columns),
                        'original_target': y,
                        'preprocessing_config': {
                            'target_column': 'attack_type',
                            'label_encoders': label_encoders,
                            'dataset_type': 'KDD_Cup_1999'
                        }
                    }
                    
                    processed_name = f"{dataset_name}_processed"
                    st.session_state.datasets[processed_name] = processed_data
                
                st.success(f"✅ KDD dataset loaded and processed: {len(df)} records ready for training!")
                st.info("🎯 **Ready for Model Training!** Go to the Model Training page to start building your anomaly detection models.")
                
                # Show basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Features", len(X.columns))  # Use processed features
                with col3:
                    normal_count = int(np.sum(y_binary == 0))
                    anomaly_count = int(np.sum(y_binary == 1))
                    st.metric("Normal/Anomaly", f"{normal_count:,}/{anomaly_count:,}")
                
                # Show attack distribution
                st.subheader("Attack Type Distribution")
                attack_counts = df['attack_type'].value_counts().head(10)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = px.bar(
                        x=attack_counts.index,
                        y=attack_counts.values,
                        title="Top 10 Attack Types",
                        labels={'x': 'Attack Type', 'y': 'Count'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Binary classification stats
                    normal_count = (df['attack_type'] == 'normal').sum()
                    attack_count = len(df) - normal_count
                    
                    st.metric("Normal Traffic", f"{normal_count:,}")
                    st.metric("Attack Traffic", f"{attack_count:,}")
                    anomaly_rate = (attack_count / len(df)) * 100
                    st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
                
                uploaded_file = None  # Set to None since we're using built-in data
                
            except Exception as e:
                st.error(f"Error loading KDD dataset: {str(e)}")
                uploaded_file = None

else:
    # File upload section
    st.header("Upload Custom Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload CSV, Excel, or JSON files containing cybersecurity data"
        )

# Initialize uploaded_file if it doesn't exist
if 'uploaded_file' not in locals():
    uploaded_file = None

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
