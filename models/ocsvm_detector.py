import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os
from datetime import datetime

class OCSDVMDetector:
    """One-Class SVM for Anomaly Detection"""
    
    def __init__(self, kernel='rbf', gamma='scale', nu=0.05):
        """
        Initialize OCSVM detector
        
        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient ('scale', 'auto', or float)
            nu: Upper bound on fraction of training errors and lower bound of support vectors
        """
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_info = {
            'training_time': None,
            'n_samples': 0,
            'n_features': 0,
            'n_support_vectors': 0,
            'outlier_fraction': 0.0
        }
    
    def preprocess_data(self, X, fit_scaler=True):
        """Preprocess features with scaling"""
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def train(self, X_train, progress_callback=None):
        """
        Train the One-Class SVM model
        
        Args:
            X_train: Training features (normal data only)
            progress_callback: Optional callback for progress updates
        """
        start_time = datetime.now()
        
        # Preprocess data
        X_train_scaled = self.preprocess_data(X_train, fit_scaler=True)
        
        if progress_callback:
            progress_callback("Preprocessing completed", 25)
        
        # Train the model
        self.model.fit(X_train_scaled)
        
        if progress_callback:
            progress_callback("Model training completed", 75)
        
        # Store training information
        end_time = datetime.now()
        self.training_info = {
            'training_time': (end_time - start_time).total_seconds(),
            'n_samples': X_train_scaled.shape[0],
            'n_features': X_train_scaled.shape[1],
            'n_support_vectors': self.model.support_vectors_.shape[0] if hasattr(self.model, 'support_vectors_') else 0,
            'outlier_fraction': self.nu
        }
        
        # Calculate decision scores on training data
        decision_scores = self.model.decision_function(X_train_scaled)
        self.training_info['decision_threshold'] = np.percentile(decision_scores, (1 - self.nu) * 100)
        
        if progress_callback:
            progress_callback("Training completed successfully", 100)
        
        self.is_fitted = True
        return self.training_info
    
    def predict(self, X, return_scores=False):
        """
        Make predictions on new data
        
        Args:
            X: Input features
            return_scores: Whether to return decision scores
            
        Returns:
            predictions: 1 for normal, -1 for anomaly (converted to 0/1 format)
            scores: Decision function scores (if return_scores=True)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.preprocess_data(X, fit_scaler=False)
        
        # Get predictions (-1 for outliers, 1 for inliers)
        raw_predictions = self.model.predict(X_scaled)
        
        # Convert to binary format (0 for normal, 1 for anomaly)
        predictions = np.where(raw_predictions == -1, 1, 0)
        
        if return_scores:
            # Get decision function scores (more negative = more anomalous)
            decision_scores = self.model.decision_function(X_scaled)
            # Convert to anomaly scores (higher = more anomalous)
            anomaly_scores = -decision_scores
            # Normalize to 0-1 range
            min_score = np.min(anomaly_scores)
            max_score = np.max(anomaly_scores)
            if max_score > min_score:
                anomaly_scores = (anomaly_scores - min_score) / (max_score - min_score)
            else:
                anomaly_scores = np.zeros_like(anomaly_scores)
            
            return predictions, anomaly_scores
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True labels (0 for normal, 1 for anomaly)
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions, scores = self.predict(X_test, return_scores=True)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y_test)
        
        # Classification report
        report = classification_report(y_test, predictions, 
                                     target_names=['Normal', 'Anomaly'], 
                                     output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # AUC score (if possible)
        try:
            auc_score = roc_auc_score(y_test, scores)
        except:
            auc_score = 0.0
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'scores': scores,
            'training_info': self.training_info
        }
    
    def get_anomaly_score(self, X):
        """Get anomaly scores for input data"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before scoring")
        
        _, scores = self.predict(X, return_scores=True)
        return scores
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'training_info': self.training_info,
            'config': {
                'kernel': self.kernel,
                'gamma': self.gamma,
                'nu': self.nu
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.training_info = model_data['training_info']
        
        # Load config
        config = model_data['config']
        self.kernel = config['kernel']
        self.gamma = config['gamma']
        self.nu = config['nu']
        
        self.is_fitted = True
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_fitted:
            return {"status": "Model not trained"}
        
        info = {
            "model_type": "One-Class SVM",
            "kernel": self.kernel,
            "gamma": self.gamma,
            "nu": self.nu,
            "is_fitted": self.is_fitted,
            **self.training_info
        }
        
        return info