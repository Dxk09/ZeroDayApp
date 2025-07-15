import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import random

class ZeroDaySimulator:
    """
    Advanced zero-day attack simulator that generates realistic attack patterns
    not seen in training data to test true zero-day detection capabilities
    """
    
    def __init__(self):
        self.attack_patterns = {
            'advanced_persistent_threat': {
                'duration': (300, 3600),  # 5 minutes to 1 hour
                'src_bytes': (100, 1000),
                'dst_bytes': (50, 500),
                'protocol_type': ['tcp'],
                'service': ['http', 'ftp', 'smtp'],
                'flag': ['SF', 'S0'],
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': (0, 2),
                'num_failed_logins': (0, 1),
                'logged_in': 1,
                'num_compromised': (0, 1),
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': (0, 1),
                'num_file_creations': (1, 5),
                'num_shells': (0, 1),
                'num_access_files': (1, 3),
                'num_outbound_cmds': (0, 2),
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': (1, 10),
                'srv_count': (1, 10),
                'serror_rate': (0.0, 0.1),
                'srv_serror_rate': (0.0, 0.1),
                'rerror_rate': (0.0, 0.1),
                'srv_rerror_rate': (0.0, 0.1),
                'same_srv_rate': (0.5, 1.0),
                'diff_srv_rate': (0.0, 0.3),
                'srv_diff_host_rate': (0.0, 0.2),
                'dst_host_count': (1, 50),
                'dst_host_srv_count': (1, 20),
                'dst_host_same_srv_rate': (0.7, 1.0),
                'dst_host_diff_srv_rate': (0.0, 0.3),
                'dst_host_same_src_port_rate': (0.0, 0.2),
                'dst_host_srv_diff_host_rate': (0.0, 0.1),
                'dst_host_serror_rate': (0.0, 0.1),
                'dst_host_srv_serror_rate': (0.0, 0.1),
                'dst_host_rerror_rate': (0.0, 0.1),
                'dst_host_srv_rerror_rate': (0.0, 0.1)
            },
            'polymorphic_malware': {
                'duration': (1, 30),
                'src_bytes': (200, 2000),
                'dst_bytes': (100, 1000),
                'protocol_type': ['tcp', 'udp'],
                'service': ['http', 'private', 'other'],
                'flag': ['SF', 'REJ'],
                'land': 0,
                'wrong_fragment': (0, 1),
                'urgent': (0, 1),
                'hot': (0, 3),
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': (0, 2),
                'num_shells': 0,
                'num_access_files': (0, 1),
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': (5, 100),
                'srv_count': (1, 50),
                'serror_rate': (0.0, 0.2),
                'srv_serror_rate': (0.0, 0.2),
                'rerror_rate': (0.0, 0.3),
                'srv_rerror_rate': (0.0, 0.3),
                'same_srv_rate': (0.2, 0.8),
                'diff_srv_rate': (0.1, 0.5),
                'srv_diff_host_rate': (0.0, 0.4),
                'dst_host_count': (10, 255),
                'dst_host_srv_count': (1, 100),
                'dst_host_same_srv_rate': (0.3, 0.9),
                'dst_host_diff_srv_rate': (0.0, 0.4),
                'dst_host_same_src_port_rate': (0.0, 0.3),
                'dst_host_srv_diff_host_rate': (0.0, 0.2),
                'dst_host_serror_rate': (0.0, 0.2),
                'dst_host_srv_serror_rate': (0.0, 0.2),
                'dst_host_rerror_rate': (0.0, 0.3),
                'dst_host_srv_rerror_rate': (0.0, 0.3)
            },
            'ai_powered_attack': {
                'duration': (10, 120),
                'src_bytes': (500, 5000),
                'dst_bytes': (250, 2500),
                'protocol_type': ['tcp'],
                'service': ['http', 'https', 'ftp'],
                'flag': ['SF'],
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': (1, 5),
                'num_failed_logins': (0, 2),
                'logged_in': (0, 1),
                'num_compromised': (0, 2),
                'root_shell': 0,
                'su_attempted': (0, 1),
                'num_root': (0, 1),
                'num_file_creations': (0, 3),
                'num_shells': (0, 1),
                'num_access_files': (0, 4),
                'num_outbound_cmds': (0, 1),
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': (3, 30),
                'srv_count': (2, 20),
                'serror_rate': (0.0, 0.05),
                'srv_serror_rate': (0.0, 0.05),
                'rerror_rate': (0.0, 0.05),
                'srv_rerror_rate': (0.0, 0.05),
                'same_srv_rate': (0.6, 0.95),
                'diff_srv_rate': (0.0, 0.2),
                'srv_diff_host_rate': (0.0, 0.15),
                'dst_host_count': (5, 100),
                'dst_host_srv_count': (2, 50),
                'dst_host_same_srv_rate': (0.8, 1.0),
                'dst_host_diff_srv_rate': (0.0, 0.2),
                'dst_host_same_src_port_rate': (0.0, 0.1),
                'dst_host_srv_diff_host_rate': (0.0, 0.1),
                'dst_host_serror_rate': (0.0, 0.05),
                'dst_host_srv_serror_rate': (0.0, 0.05),
                'dst_host_rerror_rate': (0.0, 0.05),
                'dst_host_srv_rerror_rate': (0.0, 0.05)
            },
            'supply_chain_attack': {
                'duration': (60, 600),
                'src_bytes': (1000, 10000),
                'dst_bytes': (500, 5000),
                'protocol_type': ['tcp'],
                'service': ['http', 'ftp', 'smtp'],
                'flag': ['SF'],
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': (2, 10),
                'num_failed_logins': 0,
                'logged_in': 1,
                'num_compromised': (1, 5),
                'root_shell': (0, 1),
                'su_attempted': (0, 1),
                'num_root': (0, 2),
                'num_file_creations': (3, 15),
                'num_shells': (0, 2),
                'num_access_files': (2, 10),
                'num_outbound_cmds': (1, 5),
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': (1, 5),
                'srv_count': (1, 5),
                'serror_rate': (0.0, 0.02),
                'srv_serror_rate': (0.0, 0.02),
                'rerror_rate': (0.0, 0.02),
                'srv_rerror_rate': (0.0, 0.02),
                'same_srv_rate': (0.8, 1.0),
                'diff_srv_rate': (0.0, 0.1),
                'srv_diff_host_rate': (0.0, 0.1),
                'dst_host_count': (1, 20),
                'dst_host_srv_count': (1, 10),
                'dst_host_same_srv_rate': (0.9, 1.0),
                'dst_host_diff_srv_rate': (0.0, 0.1),
                'dst_host_same_src_port_rate': (0.0, 0.1),
                'dst_host_srv_diff_host_rate': (0.0, 0.05),
                'dst_host_serror_rate': (0.0, 0.02),
                'dst_host_srv_serror_rate': (0.0, 0.02),
                'dst_host_rerror_rate': (0.0, 0.02),
                'dst_host_srv_rerror_rate': (0.0, 0.02)
            }
        }
        
        # Feature names matching KDD Cup dataset
        self.feature_names = [
            'duration', 'src_bytes', 'dst_bytes', 'protocol_type', 'service', 'flag',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
        # Encoding mappings for categorical features
        self.protocol_encodings = {'tcp': 0, 'udp': 1, 'icmp': 2}
        self.service_encodings = {
            'http': 0, 'smtp': 1, 'ftp': 2, 'private': 3, 'other': 4, 
            'https': 5, 'telnet': 6, 'ssh': 7, 'domain': 8, 'finger': 9
        }
        self.flag_encodings = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5}
    
    def generate_zero_day_attack(self, attack_type, num_samples=100):
        """Generate realistic zero-day attack samples"""
        if attack_type not in self.attack_patterns:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        pattern = self.attack_patterns[attack_type]
        samples = []
        
        for _ in range(num_samples):
            sample = {}
            
            for feature in self.feature_names:
                if feature in pattern:
                    value_spec = pattern[feature]
                    
                    if isinstance(value_spec, tuple):
                        # Numerical range
                        if isinstance(value_spec[0], int):
                            sample[feature] = random.randint(value_spec[0], value_spec[1])
                        else:
                            sample[feature] = random.uniform(value_spec[0], value_spec[1])
                    
                    elif isinstance(value_spec, list):
                        # Categorical choices
                        choice = random.choice(value_spec)
                        if feature == 'protocol_type':
                            sample[feature] = self.protocol_encodings.get(choice, 0)
                        elif feature == 'service':
                            sample[feature] = self.service_encodings.get(choice, 0)
                        elif feature == 'flag':
                            sample[feature] = self.flag_encodings.get(choice, 0)
                        else:
                            sample[feature] = choice
                    
                    else:
                        # Fixed value
                        sample[feature] = value_spec
                else:
                    # Default values for features not specified
                    sample[feature] = 0
            
            samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def generate_evasion_variants(self, base_sample, num_variants=10):
        """Generate evasion variants of an attack to test robustness"""
        variants = []
        
        for _ in range(num_variants):
            variant = base_sample.copy()
            
            # Apply evasion techniques
            if random.random() < 0.3:  # Time-based evasion
                variant['duration'] = max(1, variant['duration'] + random.randint(-5, 5))
            
            if random.random() < 0.3:  # Size-based evasion
                variant['src_bytes'] = max(0, variant['src_bytes'] + random.randint(-100, 100))
                variant['dst_bytes'] = max(0, variant['dst_bytes'] + random.randint(-50, 50))
            
            if random.random() < 0.2:  # Protocol confusion
                variant['protocol_type'] = random.choice(list(self.protocol_encodings.values()))
            
            if random.random() < 0.2:  # Service hopping
                variant['service'] = random.choice(list(self.service_encodings.values()))
            
            if random.random() < 0.1:  # Connection state manipulation
                variant['flag'] = random.choice(list(self.flag_encodings.values()))
            
            # Add noise to statistical features
            for feature in ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate']:
                if feature in variant:
                    noise = random.uniform(-0.01, 0.01)
                    variant[feature] = max(0, min(1, variant[feature] + noise))
            
            variants.append(variant)
        
        return pd.DataFrame(variants)
    
    def create_zero_day_test_suite(self, samples_per_attack=50):
        """Create comprehensive zero-day test suite"""
        test_data = []
        attack_labels = []
        
        for attack_type in self.attack_patterns.keys():
            # Generate base attacks
            base_attacks = self.generate_zero_day_attack(attack_type, samples_per_attack)
            
            # Generate evasion variants for some samples
            evasion_samples = []
            for idx in range(min(10, len(base_attacks))):
                variants = self.generate_evasion_variants(base_attacks.iloc[idx], 5)
                evasion_samples.append(variants)
            
            if evasion_samples:
                evasion_df = pd.concat(evasion_samples, ignore_index=True)
                combined_attacks = pd.concat([base_attacks, evasion_df], ignore_index=True)
            else:
                combined_attacks = base_attacks
            
            test_data.append(combined_attacks)
            attack_labels.extend([attack_type] * len(combined_attacks))
        
        # Combine all attack types
        zero_day_data = pd.concat(test_data, ignore_index=True)
        
        # Ensure all features are present and properly ordered
        for feature in self.feature_names:
            if feature not in zero_day_data.columns:
                zero_day_data[feature] = 0
        
        zero_day_data = zero_day_data[self.feature_names]
        
        return zero_day_data, attack_labels
    
    def get_attack_descriptions(self):
        """Get descriptions of each zero-day attack type"""
        descriptions = {
            'advanced_persistent_threat': 'Long-duration stealth attack maintaining persistent access',
            'polymorphic_malware': 'Self-modifying malware that changes its signature',
            'ai_powered_attack': 'Machine learning-driven attack adapting to defenses',
            'supply_chain_attack': 'Compromise through trusted third-party components'
        }
        return descriptions