import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class KDDZeroDayFilter:
    """
    Filter KDD dataset to separate known attacks from zero-day attacks
    for realistic zero-day detection evaluation
    """
    
    def __init__(self):
        # Known attack types present in KDDTrain+
        self.known_attacks = {
            'back', 'neptune', 'satan', 'smurf', 'teardrop', 'warezclient', 
            'warezmaster', 'buffer_overflow', 'ftp_write', 'guess_passwd',
            'imap', 'ipsweep', 'land', 'loadmodule', 'multihop', 'nmap',
            'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'spy'
        }
        
        # Zero-day attack types (present in KDDTest+ but not in KDDTrain+)
        self.zero_day_attacks = {
            'apache2', 'mailbomb', 'processtable', 'snmpgetattack', 'snmpguess',
            'mscan', 'httptunnel', 'worm', 'sendmail', 'xlock', 'xsnoop', 
            'named', 'saint', 'udpstorm', 'ps'
        }
        
        # Attack categories for analysis
        self.attack_categories = {
            'DOS': ['back', 'neptune', 'smurf', 'teardrop', 'pod', 'apache2', 'mailbomb', 'processtable', 'udpstorm'],
            'Probe': ['satan', 'ipsweep', 'nmap', 'portsweep', 'saint', 'mscan'],
            'R2L': ['warezclient', 'warezmaster', 'ftp_write', 'guess_passwd', 'imap', 'phf', 'spy', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'sendmail'],
            'U2R': ['buffer_overflow', 'loadmodule', 'multihop', 'perl', 'rootkit', 'ps', 'httptunnel', 'worm', 'named']
        }
    
    def filter_training_data(self, kdd_train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter KDDTrain+ to only include known attack types and normal traffic
        """
        # Include normal traffic and known attacks only
        valid_labels = self.known_attacks.union({'normal'})
        
        filtered_df = kdd_train_df[kdd_train_df['attack_type'].isin(valid_labels)].copy()
        
        print(f"Training data filtered: {len(filtered_df)} samples from {len(kdd_train_df)} total")
        print(f"Attack types in training: {sorted(filtered_df['attack_type'].unique())}")
        
        return filtered_df
    
    def filter_zero_day_test_data(self, kdd_test_df: pd.DataFrame, include_normal: bool = True) -> pd.DataFrame:
        """
        Filter KDDTest+ to only include zero-day attacks and optionally normal traffic
        """
        if include_normal:
            valid_labels = self.zero_day_attacks.union({'normal'})
        else:
            valid_labels = self.zero_day_attacks
        
        filtered_df = kdd_test_df[kdd_test_df['attack_type'].isin(valid_labels)].copy()
        
        print(f"Zero-day test data filtered: {len(filtered_df)} samples from {len(kdd_test_df)} total")
        print(f"Zero-day attack types: {sorted(filtered_df[filtered_df['attack_type'] != 'normal']['attack_type'].unique())}")
        
        if include_normal:
            normal_count = len(filtered_df[filtered_df['attack_type'] == 'normal'])
            attack_count = len(filtered_df[filtered_df['attack_type'] != 'normal'])
            print(f"Normal samples: {normal_count}, Zero-day attacks: {attack_count}")
        
        return filtered_df
    
    def analyze_zero_day_distribution(self, zero_day_df: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of zero-day attacks by category
        """
        analysis = {
            'total_samples': len(zero_day_df),
            'attack_types': {},
            'categories': {},
            'zero_day_only': {}
        }
        
        # Count by attack type
        attack_counts = zero_day_df['attack_type'].value_counts()
        analysis['attack_types'] = attack_counts.to_dict()
        
        # Count by category
        for category, attacks in self.attack_categories.items():
            category_count = len(zero_day_df[zero_day_df['attack_type'].isin(attacks)])
            if category_count > 0:
                analysis['categories'][category] = category_count
        
        # Zero-day attacks only (excluding normal)
        zero_day_only_df = zero_day_df[zero_day_df['attack_type'] != 'normal']
        analysis['zero_day_only'] = {
            'total': len(zero_day_only_df),
            'types': zero_day_only_df['attack_type'].value_counts().to_dict()
        }
        
        return analysis
    
    def prepare_binary_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare binary labels for anomaly detection (0=normal, 1=anomaly)
        """
        # Separate features and labels
        X = df.drop('attack_type', axis=1)
        y = (df['attack_type'] != 'normal').astype(int)
        
        return X, y
    
    def get_attack_info(self, attack_type: str) -> Dict:
        """
        Get information about a specific attack type
        """
        info = {
            'name': attack_type,
            'is_zero_day': attack_type in self.zero_day_attacks,
            'is_known': attack_type in self.known_attacks,
            'category': None
        }
        
        # Find category
        for category, attacks in self.attack_categories.items():
            if attack_type in attacks:
                info['category'] = category
                break
        
        return info
    
    def create_zero_day_challenge(self, kdd_train_df: pd.DataFrame, kdd_test_df: pd.DataFrame) -> Dict:
        """
        Create a complete zero-day detection challenge dataset
        """
        # Filter training data (known attacks + normal)
        train_df = self.filter_training_data(kdd_train_df)
        
        # Filter test data (zero-day attacks + normal)
        test_df = self.filter_zero_day_test_data(kdd_test_df, include_normal=True)
        
        # Prepare features and labels
        X_train, y_train = self.prepare_binary_labels(train_df)
        X_test, y_test = self.prepare_binary_labels(test_df)
        
        # Analysis
        train_analysis = {
            'total_samples': len(train_df),
            'normal_samples': len(train_df[train_df['attack_type'] == 'normal']),
            'attack_samples': len(train_df[train_df['attack_type'] != 'normal']),
            'attack_types': sorted(train_df[train_df['attack_type'] != 'normal']['attack_type'].unique())
        }
        
        test_analysis = self.analyze_zero_day_distribution(test_df)
        
        challenge_data = {
            'train': {
                'features': X_train,
                'labels': y_train,
                'original_labels': train_df['attack_type'],
                'analysis': train_analysis
            },
            'test': {
                'features': X_test,
                'labels': y_test,
                'original_labels': test_df['attack_type'],
                'analysis': test_analysis
            },
            'zero_day_attacks': list(self.zero_day_attacks),
            'known_attacks': list(self.known_attacks)
        }
        
        return challenge_data
    
    def evaluate_zero_day_detection(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   original_labels: pd.Series) -> Dict:
        """
        Evaluate zero-day detection performance with detailed metrics
        """
        results = {
            'overall': {
                'accuracy': float(np.mean(y_true == y_pred)),
                'total_samples': len(y_true),
                'true_positives': int(np.sum((y_true == 1) & (y_pred == 1))),
                'true_negatives': int(np.sum((y_true == 0) & (y_pred == 0))),
                'false_positives': int(np.sum((y_true == 0) & (y_pred == 1))),
                'false_negatives': int(np.sum((y_true == 1) & (y_pred == 0)))
            },
            'zero_day_only': {},
            'by_attack_type': {}
        }
        
        # Calculate derived metrics
        tp = results['overall']['true_positives']
        tn = results['overall']['true_negatives']
        fp = results['overall']['false_positives']
        fn = results['overall']['false_negatives']
        
        results['overall']['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        results['overall']['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results['overall']['f1_score'] = 2 * (results['overall']['precision'] * results['overall']['recall']) / (results['overall']['precision'] + results['overall']['recall']) if (results['overall']['precision'] + results['overall']['recall']) > 0 else 0.0
        
        # Zero-day attacks only (excluding normal traffic)
        zero_day_mask = original_labels != 'normal'
        if np.any(zero_day_mask):
            zero_day_true = y_true[zero_day_mask]
            zero_day_pred = y_pred[zero_day_mask]
            
            results['zero_day_only'] = {
                'total_zero_day_attacks': int(np.sum(zero_day_mask)),
                'detected_zero_day_attacks': int(np.sum(zero_day_pred == 1)),
                'zero_day_detection_rate': float(np.mean(zero_day_pred == 1)),
                'missed_zero_day_attacks': int(np.sum(zero_day_pred == 0))
            }
        
        # By attack type
        for attack_type in original_labels.unique():
            if attack_type != 'normal':
                attack_mask = original_labels == attack_type
                if np.any(attack_mask):
                    attack_true = y_true[attack_mask]
                    attack_pred = y_pred[attack_mask]
                    
                    results['by_attack_type'][attack_type] = {
                        'total_samples': int(np.sum(attack_mask)),
                        'detected_samples': int(np.sum(attack_pred == 1)),
                        'detection_rate': float(np.mean(attack_pred == 1)),
                        'is_zero_day': attack_type in self.zero_day_attacks
                    }
        
        return results