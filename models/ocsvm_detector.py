import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import pickle
import os

class OCSVMDetector:
    """One-Class Support Vector Machine for Zero-Day Anomaly Detection"""
    
    def __init__(self, kernel='rbf', nu=0.1, gamma='scale'):
        """
        Initialize OCSVM detector
        
        Parameters:
        - kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
        - nu: Upper bound on fraction of training errors and lower bound of support vectors
        - gamma: Kernel coefficient
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_stats = {}
        
    def preprocess_data(self, X, fit_scaler=True):
        """Preprocess features with scaling"""
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def train(self, X_train, y_train=None, progress_callback=None):
        """
        Train OCSVM on normal traffic only
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels (used to filter normal traffic)
        - progress_callback: Optional callback for progress updates
        """
        
        # Filter to normal traffic only for OCSVM training
        if y_train is not None:
            normal_indices = y_train == 0  # 0 = normal traffic
            X_normal = X_train[normal_indices]
            if progress_callback:
                progress_callback(f"Filtered to {len(X_normal)} normal samples from {len(X_train)} total")
        else:
            X_normal = X_train
            
        # Preprocess data
        X_scaled = self.preprocess_data(X_normal, fit_scaler=True)
        
        if progress_callback:
            progress_callback("Training OCSVM on normal traffic patterns...")
            
        # Train OCSVM
        self.model.fit(X_scaled)
        
        # Store training statistics
        self.training_stats = {
            'n_support_vectors': int(self.model.n_support_),
            'support_vector_ratio': float(self.model.n_support_ / len(X_scaled)),
            'normal_samples_used': int(len(X_scaled)),
            'total_samples_available': int(len(X_train) if y_train is not None else len(X_train))
        }
        
        self.is_fitted = True
        
        if progress_callback:
            progress_callback(f"OCSVM trained with {int(self.model.n_support_)} support vectors")
            
        return self.training_stats
    
    def predict(self, X, return_scores=False):
        """
        Make predictions on new data
        
        Parameters:
        - X: Features to predict
        - return_scores: If True, return decision scores along with predictions
        
        Returns:
        - predictions: 1 for normal, -1 for anomaly (OCSVM format)
        - scores: Decision function scores (if return_scores=True)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.preprocess_data(X, fit_scaler=False)
        
        # Get predictions and scores
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        if return_scores:
            return predictions, scores
        return predictions
    
    def predict_binary(self, X, return_probabilities=False):
        """
        Make binary predictions compatible with binary classification format
        
        Returns:
        - predictions: 0 for normal, 1 for anomaly
        - probabilities: Normalized scores (if return_probabilities=True)
        """
        predictions, scores = self.predict(X, return_scores=True)
        
        # Convert OCSVM format (-1, 1) to binary format (0, 1)
        binary_predictions = (predictions == -1).astype(int)
        
        if return_probabilities:
            # Normalize scores to [0, 1] range for probability-like interpretation
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            # Invert scores so higher values indicate higher anomaly probability
            anomaly_probabilities = 1 - normalized_scores
            return binary_predictions, anomaly_probabilities
            
        return binary_predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions, probabilities = self.predict_binary(X_test, return_probabilities=True)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y_test)
        
        # Handle case where all predictions are one class
        try:
            auc_score = roc_auc_score(y_test, probabilities)
        except ValueError:
            auc_score = 0.5  # Default AUC for constant predictions
        
        # Classification report
        report = classification_report(y_test, predictions, 
                                     target_names=['Normal', 'Anomaly'], 
                                     output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities,
            'decision_scores': self.predict(X_test, return_scores=True)[1]
        }
    
    def get_anomaly_score(self, X):
        """Get anomaly scores for input data"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting anomaly scores")
        
        _, scores = self.predict(X, return_scores=True)
        # Convert to anomaly scores (negative decision function values indicate anomalies)
        anomaly_scores = -scores
        return anomaly_scores
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Cannot save model that hasn't been trained")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'kernel': self.kernel,
            'nu': self.nu,
            'gamma': self.gamma,
            'training_stats': self.training_stats,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.kernel = model_data['kernel']
        self.nu = model_data['nu']
        self.gamma = model_data['gamma']
        self.training_stats = model_data['training_stats']
        self.is_fitted = model_data['is_fitted']
    
    def optimize_hyperparameters(self, X_train, y_train, param_grid=None, cv=3):
        """
        Optimize OCSVM hyperparameters using cross-validation
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - param_grid: Parameter grid for grid search
        - cv: Cross-validation folds
        """
        if param_grid is None:
            param_grid = {
                'nu': [0.01, 0.05, 0.1, 0.2, 0.3],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        
        # Filter to normal traffic only
        normal_indices = y_train == 0
        X_normal = X_train[normal_indices]
        X_scaled = self.preprocess_data(X_normal, fit_scaler=True)
        
        # Simple approach: try each parameter combination manually
        best_score = float('-inf')
        best_params = None
        
        # Iterate through parameter combinations
        for kernel in param_grid['kernel']:
            for nu in param_grid['nu']:
                for gamma in param_grid['gamma']:
                    try:
                        # Create and fit model
                        temp_model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
                        temp_model.fit(X_scaled)
                        
                        # Score based on decision function scores
                        scores = temp_model.decision_function(X_scaled)
                        score = np.mean(scores)  # Higher mean score is better
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'kernel': kernel, 'nu': nu, 'gamma': gamma}
                    except:
                        continue  # Skip invalid parameter combinations
        
        # Update model with best parameters
        if best_params:
            self.kernel = best_params['kernel']
            self.nu = best_params['nu']
            self.gamma = best_params['gamma']
            self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
            
            # Retrain with best parameters
            self.model.fit(X_scaled)
            self.is_fitted = True
            
            # Update training stats
            self.training_stats = {
                'n_support_vectors': int(self.model.n_support_),
                'support_vector_ratio': float(self.model.n_support_ / len(X_scaled)),
                'normal_samples_used': int(len(X_scaled)),
                'total_samples_available': int(len(X_train))
            }
        else:
            # Fallback to default parameters
            self.model = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
            self.model.fit(X_scaled)
            self.is_fitted = True
            best_params = {'kernel': 'rbf', 'nu': 0.1, 'gamma': 'scale'}
            best_score = 0.0
            
            # Update training stats for fallback case
            self.training_stats = {
                'n_support_vectors': int(self.model.n_support_),
                'support_vector_ratio': float(self.model.n_support_ / len(X_scaled)),
                'normal_samples_used': int(len(X_scaled)),
                'total_samples_available': int(len(X_train))
            }
        
        return best_params, best_score