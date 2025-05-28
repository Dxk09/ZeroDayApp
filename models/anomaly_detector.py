import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os

class CyberSecurityDataset(Dataset):
    """Custom dataset class for cybersecurity data"""
    
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class AnomalyDetector(nn.Module):
    """Deep Neural Network for Anomaly Detection"""
    
    def __init__(self, num_input_units, num_hidden_layers=4, num_units_per_hidden_layer=4):
        super(AnomalyDetector, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(num_input_units, num_units_per_hidden_layer))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(num_units_per_hidden_layer, num_units_per_hidden_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer (binary classification)
        layers.append(nn.Linear(num_units_per_hidden_layer, 1))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AnomalyDetectionModel:
    """Complete anomaly detection model with training and prediction capabilities"""
    
    def __init__(self, num_input_units, num_hidden_layers=4, num_units_per_hidden_layer=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AnomalyDetector(num_input_units, num_hidden_layers, num_units_per_hidden_layer)
        self.model.to(self.device)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epochs': []
        }
        
    def preprocess_data(self, X, fit_scaler=True):
        """Preprocess features with scaling"""
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              num_epochs=1000, batch_size=32, learning_rate=0.001, 
              patience=50, progress_callback=None):
        """Train the anomaly detection model"""
        
        # Preprocess data
        X_train_scaled = self.preprocess_data(X_train, fit_scaler=True)
        
        # Create datasets
        train_dataset = CyberSecurityDataset(X_train_scaled, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.preprocess_data(X_val, fit_scaler=False)
            val_dataset = CyberSecurityDataset(X_val_scaled, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Define loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epochs': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            val_loss = 0.0
            val_accuracy = 0.0
            
            if X_val is not None:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(self.device), labels.to(self.device)
                        outputs = self.model(features).squeeze()
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        predicted = (outputs > 0.5).float()
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            else:
                avg_val_loss = 0.0
                val_accuracy = 0.0
            
            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['epochs'].append(epoch + 1)
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, num_epochs, avg_train_loss, avg_val_loss, 
                                train_accuracy, val_accuracy)
        
        self.is_fitted = True
        return self.training_history
    
    def predict(self, X, return_probabilities=False):
        """Make predictions on new data"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.preprocess_data(X, fit_scaler=False)
        dataset = CyberSecurityDataset(X_scaled)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for features in data_loader:
                if isinstance(features, tuple):
                    features = features[0]
                features = features.to(self.device)
                outputs = self.model(features).squeeze()
                
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()
                
                probabilities.extend(probs)
                predictions.extend(preds)
        
        if return_probabilities:
            return np.array(predictions), np.array(probabilities)
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions, probabilities = self.predict(X_test, return_probabilities=True)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y_test)
        auc_score = roc_auc_score(y_test, probabilities)
        
        # Classification report
        report = classification_report(y_test, predictions, 
                                     target_names=['Normal', 'Anomaly'], 
                                     output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'training_history': self.training_history,
            'model_config': {
                'num_input_units': self.model.network[0].in_features,
                'num_hidden_layers': len([l for l in self.model.network if isinstance(l, nn.Linear)]) - 2,
                'num_units_per_hidden_layer': self.model.network[0].out_features
            }
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Reconstruct model with saved config
        config = checkpoint['model_config']
        self.model = AnomalyDetector(
            config['num_input_units'],
            config['num_hidden_layers'],
            config['num_units_per_hidden_layer']
        )
        self.model.to(self.device)
        
        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.training_history = checkpoint['training_history']
        self.is_fitted = True
    
    def get_anomaly_score(self, X):
        """Get anomaly scores for input data"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before scoring")
        
        _, probabilities = self.predict(X, return_probabilities=True)
        return probabilities
