import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import streamlit as st

class CyberSecurityDataProcessor:
    """Data preprocessing utilities for cybersecurity datasets"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.target_column = None
        
    def load_data(self, file, file_type='csv', sheet_name=None):
        """Load data from various file formats"""
        try:
            if file_type.lower() == 'csv':
                df = pd.read_csv(file)
            elif file_type.lower() in ['xlsx', 'xls', 'excel']:
                df = pd.read_excel(file, sheet_name=sheet_name)
            elif file_type.lower() == 'json':
                df = pd.read_json(file)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def analyze_dataset(self, df):
        """Analyze dataset and provide insights"""
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'unique_values': {col: df[col].nunique() for col in df.columns},
        }
        
        # Basic statistics for numeric columns
        if analysis['numeric_columns']:
            analysis['statistics'] = df[analysis['numeric_columns']].describe().to_dict()
        
        # Check for potential target columns
        potential_targets = []
        for col in df.columns:
            if df[col].nunique() == 2 and col.lower() in ['label', 'target', 'class', 'anomaly', 'attack', 'malware']:
                potential_targets.append(col)
        analysis['potential_targets'] = potential_targets
        
        return analysis
    
    def preprocess_features(self, df, target_column=None, handle_missing='mean', 
                          encode_categorical=True, scale_features=True, 
                          scaling_method='standard'):
        """Comprehensive feature preprocessing"""
        
        processed_df = df.copy()
        
        # Separate features and target
        if target_column:
            self.target_column = target_column
            X = processed_df.drop(columns=[target_column])
            y = processed_df[target_column]
        else:
            X = processed_df
            y = None
        
        # Handle missing values
        if handle_missing:
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            if len(numeric_columns) > 0:
                if handle_missing == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                elif handle_missing == 'median':
                    imputer = SimpleImputer(strategy='median')
                else:  # mode
                    imputer = SimpleImputer(strategy='most_frequent')
                
                X[numeric_columns] = imputer.fit_transform(X[numeric_columns])
            
            if len(categorical_columns) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_columns] = cat_imputer.fit_transform(X[categorical_columns])
        
        # Encode categorical variables
        if encode_categorical:
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
        
        # Scale features
        if scale_features:
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()  # default
            
            self.scalers['features'] = scaler
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        self.feature_names = list(X.columns)
        
        return X, y
    
    def prepare_anomaly_labels(self, y, anomaly_indicators=None):
        """Prepare labels for anomaly detection"""
        if anomaly_indicators is None:
            # Default anomaly indicators
            anomaly_indicators = ['attack', 'anomaly', 'malware', 'intrusion', '1', 1, True]
        
        # Convert to binary labels (1 for anomaly, 0 for normal)
        if y.dtype == 'object':
            # String labels
            y_binary = y.str.lower().isin([str(ind).lower() for ind in anomaly_indicators]).astype(int)
        else:
            # Numeric labels
            y_binary = y.isin(anomaly_indicators).astype(int)
        
        return y_binary
    
    def create_train_val_split(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """Create train/validation/test splits"""
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_importance_data(self, X, y):
        """Calculate basic feature importance metrics"""
        from sklearn.feature_selection import mutual_info_classif
        from scipy.stats import pearsonr
        
        importance_data = {}
        
        # Mutual information
        mi_scores = mutual_info_classif(X, y)
        importance_data['mutual_info'] = dict(zip(self.feature_names, mi_scores))
        
        # Correlation with target
        correlations = []
        for i, feature in enumerate(self.feature_names):
            try:
                corr, _ = pearsonr(X.iloc[:, i], y)
                correlations.append(abs(corr))
            except:
                correlations.append(0)
        
        importance_data['correlation'] = dict(zip(self.feature_names, correlations))
        
        return importance_data
    
    def detect_anomalies_statistical(self, X, method='zscore', threshold=3):
        """Simple statistical anomaly detection"""
        
        if method == 'zscore':
            z_scores = np.abs((X - X.mean()) / X.std())
            anomalies = (z_scores > threshold).any(axis=1)
        elif method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            anomalies = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return anomalies
    
    def generate_data_quality_report(self, df):
        """Generate comprehensive data quality report"""
        
        report = {
            'overview': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'duplicates': df.duplicated().sum()
            },
            'completeness': {},
            'consistency': {},
            'validity': {}
        }
        
        # Completeness check
        for col in df.columns:
            missing = df[col].isnull().sum()
            report['completeness'][col] = {
                'missing_count': missing,
                'missing_percentage': (missing / len(df)) * 100,
                'completeness_score': ((len(df) - missing) / len(df)) * 100
            }
        
        # Consistency check (data types)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in string columns
                try:
                    pd.to_numeric(df[col], errors='raise')
                    report['consistency'][col] = 'Could be numeric'
                except:
                    report['consistency'][col] = 'Text data'
            else:
                report['consistency'][col] = 'Numeric data'
        
        # Validity check (outliers for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            report['validity'][col] = {
                'outlier_count': outliers,
                'outlier_percentage': (outliers / len(df)) * 100
            }
        
        return report
